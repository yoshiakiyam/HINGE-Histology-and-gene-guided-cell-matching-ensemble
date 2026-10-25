
import numpy as np
from scipy.sparse import issparse
import scanpy as sc
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize as sk_normalize


def _to_array(x):
        if hasattr(x, 'X'):
            arr = x.X.toarray() if issparse(x.X) else np.array(x.X)
        else:
            arr = np.array(x)
        return arr.astype(np.float32)
# ─────────────────────────────────────────────────────────────────────────────
#  Morphology-based cell type clustering + label transfer
# ─────────────────────────────────────────────────────────────────────────────

def compute_morph_cluster_labels(
    morph_panel1_raw,
    morph_panel2_raw,
    n_pca_comps           = 50,
    leiden_resolution     = 1.0,
    n_neighbors_transfer  = 15,
    min_cluster_size      = 10,
    verbose               = True,
):
    """
    Cluster Panel 1 cells by UNI morphology, transfer labels to Panel 2.

    Input expectations
    ------------------
    morph_panel1_raw / morph_panel2_raw :
        AnnData or ndarray of RAW UNI embeddings (2048-dim, L2-normalized).
        Do NOT pass count data — these features are already in a scaled space.
        normalize_total / log1p are intentionally skipped.
    """
    
    def _to_array(x):
        if hasattr(x, 'X'):
            arr = x.X.toarray() if issparse(x.X) else np.array(x.X)
        else:
            arr = np.array(x)
        return arr.astype(np.float32)

    X_p1 = _to_array(morph_panel1_raw)   # (n1, 2048)
    X_p2 = _to_array(morph_panel2_raw)   # (n2, 2048)

    # ── Sanity check ──────────────────────────────────────────────────────────
    if np.isnan(X_p1).any() or np.isnan(X_p2).any():
        raise ValueError(
            "compute_morph_cluster_labels: NaN detected in input features. "
            "Ensure morph_panel1/2 are valid L2-normalized UNI embeddings."
        )

    # ── Step 1: Cluster Panel 1 directly on UNI PCA ───────────────────────────
    # UNI features are already L2-normalized → apply PCA directly.
    # No normalize_total or log1p — those are for raw count matrices.
    adata_morph = sc.AnnData(X=X_p1.copy())

    n_comps_safe = min(n_pca_comps, X_p1.shape[1] - 1, X_p1.shape[0] - 1)
    sc.pp.pca(adata_morph, n_comps=n_comps_safe)         # no HVG, no log
    sc.pp.neighbors(adata_morph, n_pcs=n_comps_safe)
    sc.tl.leiden(
        adata_morph,
        resolution   = leiden_resolution,
        key_added    = 'morph_cluster',
        flavor       = 'igraph',
        n_iterations = 2,
        directed     = False,
    )

    raw_labels = adata_morph.obs['morph_cluster'].values.astype(str)
    unique_raw = np.unique(raw_labels)
    K_raw      = len(unique_raw)

    # ── Step 2: Merge tiny clusters ───────────────────────────────────────────
    sizes    = pd.Series(raw_labels).value_counts()
    tiny     = sizes[sizes < min_cluster_size].index.tolist()
    label_map = {c: c for c in unique_raw}

    if tiny:
        pca_coords = np.array(adata_morph.obsm['X_pca'])
        centroids  = {
            c: pca_coords[raw_labels == c].mean(axis=0)
            for c in unique_raw
        }
        non_tiny = [c for c in unique_raw if c not in tiny]
        for tc in tiny:
            if not non_tiny:
                break
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
    K             = len(unique_merged)

    label_encoder = {c: i for i, c in enumerate(sorted(unique_merged))}
    labels_p1     = np.array(
        [label_encoder[l] for l in labels_p1_str], dtype=int
    )

    if verbose:
        sizes_merged = pd.Series(labels_p1).value_counts().sort_values()
        print(f"\n  Panel 1 UNI clustering  (resolution={leiden_resolution})")
        print(f"    Raw clusters   : {K_raw}")
        print(f"    After merging  : {K}")
        print(f"    Cluster sizes  : "
              f"min={sizes_merged.min()}, "
              f"median={int(sizes_merged.median())}, "
              f"max={sizes_merged.max()}")

    # ── Step 3: Transfer labels to Panel 2 via cosine KNN ────────────────────
    # L2-normalize before KNN — features are already ~unit norm but
    # re-normalizing is safe and ensures cosine distances are correct.
    # No log1p here — values can be negative, log1p would produce NaN.
    p1_norm = sk_normalize(X_p1.astype(np.float64))
    p2_norm = sk_normalize(X_p2.astype(np.float64))

    knn = KNeighborsClassifier(
        n_neighbors = min(n_neighbors_transfer, len(labels_p1)),
        metric      = 'cosine',
        n_jobs      = -1,
    )
    knn.fit(p1_norm, labels_p1)
    labels_p2 = knn.predict(p2_norm)

    if verbose:
        # Integer labels 0..K-1 (matches labels_p1 / labels_p2 dtype)
        dist_p1 = (pd.Series(labels_p1).value_counts(normalize=True)
                     .reindex(range(K), fill_value=0.0))
        dist_p2 = (pd.Series(labels_p2).value_counts(normalize=True)
                     .reindex(range(K), fill_value=0.0))

        max_delta = 0.0
        print(f"\n  Label transfer  (KNN k={n_neighbors_transfer}, cosine)")
        print(f"  {'Cluster':>8} {'P1 frac':>10} {'P2 frac':>10} {'Δ':>8}")
        for c in range(K):
            f1    = dist_p1[c]
            f2    = dist_p2[c]
            delta = abs(f1 - f2)
            max_delta = max(max_delta, delta)
            flag  = "  <-- check" if delta > 0.05 else ""
            print(f"  {c:>8} {f1:>10.3f} {f2:>10.3f} {delta:>8.3f}{flag}")
        print(f"\n  Max fraction imbalance : {max_delta:.3f}")

    cluster_info = {
        'K'            : K,
        'labels_p1'    : labels_p1,
        'labels_p2'    : labels_p2,
        'label_encoder': label_encoder,
        'K_raw'        : K_raw,
    }

    return labels_p1, labels_p2, cluster_info






def compute_morph_cluster_labels_joint(
    morph_panel1_raw,
    morph_panel2_raw,
    n_pca_comps      = 50,
    leiden_resolution = 1.0,
    min_cluster_size  = 10,
    verbose           = True,
):
    """
    Cluster Panel 1 and Panel 2 UNI morphology features JOINTLY.

    concatenates both panels, runs PCA + Leiden on the joint graph,
    then splits the resulting labels back into P1 and P2.

    Advantages over label-transfer (HINGE4):
    - No KNN approximation error at cluster boundaries
    - P2 cells are assigned directly, not estimated from P1 neighbours
    - P1/P2 cells that co-embed naturally share the same cluster label
    - Works correctly even when n1 ≠ n2 (e.g. after dropout)

    Inputs
    ------
    morph_panel1_raw / morph_panel2_raw :
        AnnData or ndarray of L2-normalised UNI embeddings (2048-dim).
        Do NOT pass raw counts — no normalize_total / log1p applied here.

    Returns
    -------
    labels_p1   : int ndarray (n1,)
    labels_p2   : int ndarray (n2,)
    cluster_info: dict with K, K_raw, label_encoder, method='joint'
    """


    X_p1 = _to_array(morph_panel1_raw)   # (n1, D)
    X_p2 = _to_array(morph_panel2_raw)   # (n2, D)
    n1, n2 = len(X_p1), len(X_p2)

    if np.isnan(X_p1).any() or np.isnan(X_p2).any():
        raise ValueError(
            "compute_morph_cluster_labels_joint: NaN detected in input. "
            "Ensure inputs are valid L2-normalised UNI embeddings."
        )

    # ── Step 1: Concatenate ───────────────────────────────────────────────────
    X_joint = np.vstack([X_p1, X_p2])            # (n1+n2, D)
    adata_joint = sc.AnnData(X=X_joint.copy())
    adata_joint.obs['panel'] = ['P1'] * n1 + ['P2'] * n2

    # ── Step 2: Joint PCA → neighbour graph → Leiden ─────────────────────────
    # UNI features are already L2-normalised → apply PCA directly (no log1p).
    n_comps_safe = min(n_pca_comps, X_joint.shape[1] - 1, X_joint.shape[0] - 1)
    sc.pp.pca(adata_joint, n_comps=n_comps_safe)
    sc.pp.neighbors(adata_joint, n_pcs=n_comps_safe)
    sc.tl.leiden(
        adata_joint,
        resolution   = leiden_resolution,
        key_added    = 'morph_cluster',
        flavor       = 'igraph',
        n_iterations = 2,
        directed     = False,
    )

    raw_labels = adata_joint.obs['morph_cluster'].values.astype(str)
    unique_raw = np.unique(raw_labels)
    K_raw      = len(unique_raw)

    # ── Step 3: Merge tiny clusters into nearest centroid ─────────────────────
    sizes    = pd.Series(raw_labels).value_counts()
    tiny     = sizes[sizes < min_cluster_size].index.tolist()
    label_map = {c: c for c in unique_raw}

    if tiny:
        pca_coords = np.array(adata_joint.obsm['X_pca'])
        centroids  = {
            c: pca_coords[raw_labels == c].mean(axis=0)
            for c in unique_raw
        }
        non_tiny = [c for c in unique_raw if c not in tiny]
        for tc in tiny:
            if not non_tiny:
                break
            dists = {
                nc: np.linalg.norm(centroids[tc] - centroids[nc])
                for nc in non_tiny
            }
            label_map[tc] = min(dists, key=dists.get)

        if verbose:
            print(f"  Merged {len(tiny)} tiny clusters "
                  f"(< {min_cluster_size} cells) into nearest neighbour")

    merged_labels = np.array([label_map[l] for l in raw_labels])
    unique_merged = np.unique(merged_labels)
    K             = len(unique_merged)

    label_encoder = {c: i for i, c in enumerate(sorted(unique_merged))}
    labels_all    = np.array([label_encoder[l] for l in merged_labels], dtype=int)

    # ── Step 4: Split back into P1 / P2 ──────────────────────────────────────
    labels_p1 = labels_all[:n1]
    labels_p2 = labels_all[n1:]

    if verbose:
        sizes_merged = pd.Series(labels_all).value_counts().sort_values()
        print(f"\n  Joint UNI clustering  (resolution={leiden_resolution})")
        print(f"    Input cells    : P1={n1}  P2={n2}  total={n1+n2}")
        print(f"    Raw clusters   : {K_raw}")
        print(f"    After merging  : {K}")
        print(f"    Cluster sizes  : "
              f"min={sizes_merged.min()}, "
              f"median={int(sizes_merged.median())}, "
              f"max={sizes_merged.max()}")

        # ── Cross-panel balance diagnostic ───────────────────────────────────
        dist_p1   = pd.Series(labels_p1).value_counts(normalize=True).sort_index()
        dist_p2   = pd.Series(labels_p2).value_counts(normalize=True).sort_index()
        max_delta = 0.0
        print(f"\n  Cross-panel cluster fractions  (joint clustering)")
        print(f"  {'Cluster':>8} {'P1 frac':>10} {'P2 frac':>10} {'Δ':>8}")
        for ci in sorted(label_encoder.values()):
            f1    = dist_p1.get(ci, 0.0)
            f2    = dist_p2.get(ci, 0.0)
            delta = abs(f1 - f2)
            max_delta = max(max_delta, delta)
            flag  = " ⚠️" if delta > 0.05 else ""
            print(f"  {ci:>8} {f1:>10.3f} {f2:>10.3f} {delta:>8.3f}{flag}")
        print(f"\n  Max fraction imbalance : {max_delta:.3f}")

        # ── Cross-panel mixing diagnostic ─────────────────────────────────────
        # For each cluster, what fraction of its cells come from P1 vs P2?
        # Perfectly mixed = n1/(n1+n2). Separation indicates batch structure.
        expected_p1_frac = n1 / (n1 + n2)
        p1_fracs = []
        for ci in sorted(label_encoder.values()):
            mask   = labels_all == ci
            p1_in  = (labels_all[mask][:n1] if False else   # trick: recount
                      np.sum(labels_p1 == ci))
            total  = np.sum(mask)
            p1_fracs.append(p1_in / total if total > 0 else expected_p1_frac)

        separation = np.std(p1_fracs)
        print(f"\n  Panel mixing (std of P1-frac across clusters): {separation:.4f}")
        print(f"  Expected P1 frac = {expected_p1_frac:.3f}  "
              f"(separation ≈ 0 = well-mixed, > 0.05 = batch structure)")

    cluster_info = {
        'K'            : K,
        'K_raw'        : K_raw,
        'labels_p1'    : labels_p1,
        'labels_p2'    : labels_p2,
        'label_encoder': label_encoder,
        'method'       : 'joint',
    }

    return labels_p1, labels_p2, cluster_info


#########################################################################################################

#Hinge 6

def compute_morph_cluster_labels_joint_harmonized(
    morph_panel1_raw,
    morph_panel2_raw,
    n_pca_comps       = 50,
    leiden_resolution = 1.0,
    min_cluster_size  = 10,
    harmony_method    = "harmony",   # "harmony" | "mean_center" | "none"
    n_harmony_iters   = 10,
    harmony_theta     = 2.0,         # higher = more aggressive correction
    verbose           = True,
):
    """
    Joint clustering with inter-panel harmonization (HINGE6).

    Pipeline
    --------
    1. Concatenate P1 + P2 UNI embeddings → joint PCA (identical to HINGE5)
    2. Apply panel harmonization to joint PCA embeddings:
         "harmony"     → Harmony batch correction (harmonypy / scanpy)
         "mean_center" → Subtract per-panel PCA mean, re-center to global mean
         "none"        → No correction (equivalent to HINGE5)
    3. Build neighbour graph on harmonized embeddings → Leiden clustering
    4. Merge tiny clusters → split labels back into P1 / P2

    Why this helps
    --------------
    In HINGE5, synthetic Panel 2 = Panel 1 + alpha-scaled noise causes panel
    separation (mixing std up to 0.40 at alpha=1.25). Harmonization removes
    this inter-panel shift so that morphologically similar cells co-cluster
    regardless of which panel they come from.

    In real data (|delta_mean| median = 0.00264), the correction is minimal
    and harmless.

    Parameters
    ----------
    harmony_method : str
        "harmony"     : preferred; requires harmonypy (pip install harmonypy)
        "mean_center" : no extra dependencies; weaker correction
        "none"        : debug/ablation — identical to HINGE5
    harmony_theta : float
        Harmony diversity penalty. Higher values force more mixing.
        Start at 2.0; increase to 3–4 if panels still separate.
    """


    X_p1 = _to_array(morph_panel1_raw)   # (n1, D)
    X_p2 = _to_array(morph_panel2_raw)   # (n2, D)
    n1, n2 = len(X_p1), len(X_p2)

    if np.isnan(X_p1).any() or np.isnan(X_p2).any():
        raise ValueError(
            "compute_morph_cluster_labels_joint_harmonized: NaN in input. "
            "Ensure inputs are valid L2-normalised UNI embeddings."
        )

    # ── Step 1: Concatenate → joint AnnData ──────────────────────────────────
    X_joint = np.vstack([X_p1, X_p2])                     # (n1+n2, D)
    adata_joint = sc.AnnData(X=X_joint.copy())
    adata_joint.obs['panel'] = ['P1'] * n1 + ['P2'] * n2

    # ── Step 2: Joint PCA (no log1p — UNI features already scaled) ───────────
    n_comps_safe = min(n_pca_comps, X_joint.shape[1] - 1, X_joint.shape[0] - 1)
    sc.pp.pca(adata_joint, n_comps=n_comps_safe)

    X_pca_raw = np.array(adata_joint.obsm['X_pca'])  # save for diagnostics

    # ── Step 3: Harmonization ─────────────────────────────────────────────────
    if harmony_method == "harmony":
        try:
            sc.external.pp.harmony_integrate(
                adata_joint,
                key     = 'panel',
                basis   = 'X_pca',
                adjusted_basis = 'X_pca_harmony',
                max_iter_harmony = n_harmony_iters,
                theta   = harmony_theta,
                random_state = 42,
            )
            X_pca_corrected = np.array(adata_joint.obsm['X_pca_harmony'])
            if verbose:
                print(f"\n  Harmony correction applied  "
                      f"(theta={harmony_theta}, iters={n_harmony_iters})")
        except Exception as e:
            print(f"  ⚠️  Harmony failed ({e}), falling back to mean_center.")
            harmony_method = "mean_center"

    if harmony_method == "mean_center":
        # Subtract per-panel mean, re-add global mean
        # This is equivalent to removing the panel centroid shift
        X_pca_raw = np.array(adata_joint.obsm['X_pca'])
        global_mean = X_pca_raw.mean(axis=0)
        mean_p1     = X_pca_raw[:n1].mean(axis=0)
        mean_p2     = X_pca_raw[n1:].mean(axis=0)

        X_pca_corrected = X_pca_raw.copy()
        X_pca_corrected[:n1] -= (mean_p1 - global_mean)
        X_pca_corrected[n1:] -= (mean_p2 - global_mean)

        if verbose:
            shift_p1 = np.linalg.norm(mean_p1 - global_mean)
            shift_p2 = np.linalg.norm(mean_p2 - global_mean)
            print(f"\n  Mean-center correction applied")
            print(f"    P1 centroid shift corrected : {shift_p1:.4f}")
            print(f"    P2 centroid shift corrected : {shift_p2:.4f}")

    elif harmony_method == "none":
        X_pca_corrected = np.array(adata_joint.obsm['X_pca'])
        if verbose:
            print("\n  No harmonization applied (harmony_method='none')")

    # ── Panel mixing diagnostic: BEFORE vs AFTER ─────────────────────────────
    if verbose:
        def _mixing_std(X, n1):
            """Compute std of P1-fraction across KNN clusters as mixing metric."""
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=15, metric='euclidean', n_jobs=-1)
            nn.fit(X)
            _, indices = nn.kneighbors(X)
            p1_fracs = []
            for idx in indices:
                p1_count = np.sum(idx < n1)
                p1_fracs.append(p1_count / len(idx))
            return np.std(p1_fracs)

        std_before = _mixing_std(X_pca_raw, n1)
        std_after  = _mixing_std(X_pca_corrected, n1)
        expected   = n1 / (n1 + n2)
        print(f"\n  Panel mixing diagnostic (KNN-based, k=15)")
        print(f"    Expected P1 frac           : {expected:.3f}")
        print(f"    Mixing std BEFORE harmony  : {std_before:.4f}")
        print(f"    Mixing std AFTER  harmony  : {std_after:.4f}")
        improvement = std_before - std_after
        flag = "✅" if improvement > 0.01 else "⚠️ minimal improvement"
        print(f"    Improvement                : {improvement:.4f}  {flag}")

    # ── Step 4: Build neighbour graph on harmonized PCA → Leiden ─────────────
    adata_joint.obsm['X_pca_harm'] = X_pca_corrected
    sc.pp.neighbors(adata_joint, use_rep='X_pca_harm', n_pcs=n_comps_safe)
    sc.tl.leiden(
        adata_joint,
        resolution   = leiden_resolution,
        key_added    = 'morph_cluster',
        flavor       = 'igraph',
        n_iterations = 2,
        directed     = False,
    )

    raw_labels = adata_joint.obs['morph_cluster'].values.astype(str)
    unique_raw = np.unique(raw_labels)
    K_raw      = len(unique_raw)

    # ── Step 5: Merge tiny clusters ───────────────────────────────────────────
    sizes    = pd.Series(raw_labels).value_counts()
    tiny     = sizes[sizes < min_cluster_size].index.tolist()
    label_map = {c: c for c in unique_raw}

    if tiny:
        pca_coords = X_pca_corrected
        centroids  = {
            c: pca_coords[raw_labels == c].mean(axis=0)
            for c in unique_raw
        }
        non_tiny = [c for c in unique_raw if c not in tiny]
        for tc in tiny:
            if not non_tiny:
                break
            dists = {
                nc: np.linalg.norm(centroids[tc] - centroids[nc])
                for nc in non_tiny
            }
            label_map[tc] = min(dists, key=dists.get)
        if verbose:
            print(f"\n  Merged {len(tiny)} tiny clusters "
                  f"(< {min_cluster_size} cells) into nearest neighbour")

    merged_labels = np.array([label_map[l] for l in raw_labels])
    unique_merged = np.unique(merged_labels)
    K             = len(unique_merged)

    label_encoder = {c: i for i, c in enumerate(sorted(unique_merged))}
    labels_all    = np.array([label_encoder[l] for l in merged_labels], dtype=int)

    # ── Step 6: Split back into P1 / P2 ──────────────────────────────────────
    labels_p1 = labels_all[:n1]
    labels_p2 = labels_all[n1:]

    if verbose:
        sizes_merged = pd.Series(labels_all).value_counts().sort_values()
        print(f"\n  Harmonized joint clustering  "
              f"(method={harmony_method}, resolution={leiden_resolution})")
        print(f"    Input cells    : P1={n1}  P2={n2}  total={n1+n2}")
        print(f"    Raw clusters   : {K_raw}")
        print(f"    After merging  : {K}")
        print(f"    Cluster sizes  : "
              f"min={sizes_merged.min()}, "
              f"median={int(sizes_merged.median())}, "
              f"max={sizes_merged.max()}")

        # Cross-panel balance per cluster
        dist_p1   = pd.Series(labels_p1).value_counts(normalize=True).sort_index()
        dist_p2   = pd.Series(labels_p2).value_counts(normalize=True).sort_index()
        max_delta = 0.0
        print(f"\n  Cross-panel cluster fractions  (after harmonization)")
        print(f"  {'Cluster':>8} {'P1 frac':>10} {'P2 frac':>10} {'Δ':>8}")
        for ci in sorted(label_encoder.values()):
            f1    = dist_p1.get(ci, 0.0)
            f2    = dist_p2.get(ci, 0.0)
            delta = abs(f1 - f2)
            max_delta = max(max_delta, delta)
            flag  = " ⚠️" if delta > 0.05 else ""
            print(f"  {ci:>8} {f1:>10.3f} {f2:>10.3f} {delta:>8.3f}{flag}")
        print(f"\n  Max fraction imbalance (post-harmony): {max_delta:.3f}")

    cluster_info = {
        'K'             : K,
        'K_raw'         : K_raw,
        'labels_p1'     : labels_p1,
        'labels_p2'     : labels_p2,
        'label_encoder' : label_encoder,
        'method'        : f'joint_harmonized_{harmony_method}',
    }

    return labels_p1, labels_p2, cluster_info