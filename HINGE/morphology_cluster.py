
import numpy as np
from scipy.sparse import issparse
import scanpy as sc
import pandas as pd



# ─────────────────────────────────────────────────────────────────────────────
#  Morphology-based cell type clustering + label transfer
# ─────────────────────────────────────────────────────────────────────────────

def compute_morph_cluster_labels(
    morph_panel1_raw,    # (n1, 2048) raw UNI AnnData or ndarray
    morph_panel2_raw,    # (n2, 2048) raw UNI AnnData or ndarray
    n_pca_comps    = 50,
    leiden_resolution = 1.0,   # higher → more clusters → better rare-type separation
    n_neighbors_transfer = 15,
    min_cluster_size = 10,     # clusters smaller than this are merged into nearest
    verbose = True,
):
    """
    Cluster Panel 1 cells by UNI morphology (high resolution),
    then transfer labels to Panel 2 via cosine KNN.

    Uses UNI features only — no spatial coordinates, no expression,
    no existing cell type labels. No circularity with spatial matching.

    Returns
    -------
    labels_p1 : (n1,) int array  — cluster index per Panel 1 cell
    labels_p2 : (n2,) int array  — transferred cluster index per Panel 2 cell
    cluster_info : dict           — sizes, transfer quality diagnostics
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import normalize as sk_normalize

    # ── Extract raw feature matrices ──────────────────────────────────────────
    def _to_array(x):
        if hasattr(x, 'X'):
            arr = x.X.toarray() if issparse(x.X) else np.array(x.X)
        else:
            arr = np.array(x)
        return arr.astype(np.float32)

    X_p1 = _to_array(morph_panel1_raw)   # (n1, 2048)
    X_p2 = _to_array(morph_panel2_raw)   # (n2, 2048)

    # ── Step 1: Cluster Panel 1 on UNI features ───────────────────────────────
    adata_morph = sc.AnnData(X=X_p1)
    sc.pp.normalize_total(adata_morph, target_sum=1e4)
    sc.pp.log1p(adata_morph)

    n_comps_safe = min(n_pca_comps, X_p1.shape[1] - 1, X_p1.shape[0] - 1)
    sc.pp.pca(adata_morph, n_comps=n_comps_safe)
    sc.pp.neighbors(adata_morph, n_pcs=n_comps_safe)
    sc.tl.leiden(
        adata_morph,
        resolution  = leiden_resolution,
        key_added   = 'morph_cluster',
        flavor      = 'igraph',
        n_iterations= 2,
        directed    = False,
    )

    raw_labels = adata_morph.obs['morph_cluster'].values.astype(str)
    unique_raw = np.unique(raw_labels)
    K_raw      = len(unique_raw)

    # ── Step 2: Merge tiny clusters into nearest neighbour cluster ────────────
    # Prevents rare-cell clusters of size 2–5 causing instability in scoring
    sizes       = pd.Series(raw_labels).value_counts()
    tiny        = sizes[sizes < min_cluster_size].index.tolist()
    label_map   = {c: c for c in unique_raw}   # identity by default

    if tiny:
        # For each tiny cluster, find the nearest non-tiny cluster
        # by comparing cluster centroids in PCA space
        pca_coords  = np.array(adata_morph.obsm['X_pca'])
        centroids   = {}
        for c in unique_raw:
            mask = raw_labels == c
            centroids[c] = pca_coords[mask].mean(axis=0)

        non_tiny = [c for c in unique_raw if c not in tiny]
        for tc in tiny:
            dists = {
                nc: np.linalg.norm(centroids[tc] - centroids[nc])
                for nc in non_tiny
            }
            label_map[tc] = min(dists, key=dists.get)

        if verbose:
            print(f"  Merged {len(tiny)} tiny clusters "
                  f"(< {min_cluster_size} cells) into nearest neighbour")

    labels_p1_str = np.array([label_map[l] for l in raw_labels])
    unique_merged = np.unique(labels_p1_str)
    K = len(unique_merged)

    # Convert to int for efficiency
    label_encoder = {c: i for i, c in enumerate(sorted(unique_merged))}
    labels_p1 = np.array([label_encoder[l] for l in labels_p1_str], dtype=int)

    if verbose:
        sizes_merged = pd.Series(labels_p1).value_counts().sort_values()
        print(f"\n  Panel 1 UNI clustering  "
              f"(resolution={leiden_resolution})")
        print(f"    Raw clusters   : {K_raw}")
        print(f"    After merging  : {K}")
        print(f"    Cluster sizes  : "
              f"min={sizes_merged.min()}, "
              f"median={int(sizes_merged.median())}, "
              f"max={sizes_merged.max()}")

    # ── Step 3: Transfer labels to Panel 2 via cosine KNN on raw UNI ─────────
    # Normalize both panels to unit vectors for cosine distance
    p1_norm = sk_normalize(np.log1p(X_p1))
    p2_norm = sk_normalize(np.log1p(X_p2))

    knn = KNeighborsClassifier(
        n_neighbors = min(n_neighbors_transfer, len(labels_p1)),
        metric      = 'cosine',
        n_jobs      = -1,
    )
    knn.fit(p1_norm, labels_p1)
    labels_p2 = knn.predict(p2_norm)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    if verbose:
        dist_p1 = pd.Series(labels_p1).value_counts(normalize=True).sort_index()
        dist_p2 = pd.Series(labels_p2).value_counts(normalize=True).sort_index()
        print(f"\n  Label transfer  (KNN k={n_neighbors_transfer}, cosine)")
        print(f"  {'Cluster':>8} {'P1 frac':>10} {'P2 frac':>10} {'Δ':>8}")
        max_delta = 0.0
        for c in sorted(unique_merged):
            f1 = dist_p1.get(c, 0.0)
            f2 = dist_p2.get(c, 0.0)
            delta = abs(f1 - f2)
            max_delta = max(max_delta, delta)
            flag = " ⚠️" if delta > 0.05 else ""
            print(f"  {c:>8} {f1:>10.3f} {f2:>10.3f} {delta:>8.3f}{flag}")

        # Transfer quality: what fraction of Panel 2 cells got the same label
        # as their KNN in Panel 1? (proxy for transfer reliability)
        print(f"\n  Max fraction imbalance : {max_delta:.3f}")

    cluster_info = {
        'K'           : K,
        'labels_p1'   : labels_p1,
        'labels_p2'   : labels_p2,
        'label_encoder': label_encoder,
        'K_raw'       : K_raw,
    }

    return labels_p1, labels_p2, cluster_info