
from scipy.spatial import cKDTree
import numpy as np
from sklearn.preprocessing import normalize
from scipy.sparse import issparse
from sklearn.decomposition import PCA



# ─────────────────────────────────────────────────────────────────────────────
#  Core: vectorized spatial graph convolution
# ─────────────────────────────────────────────────────────────────────────────

def graph_smooth_features(
    xy,
    features,
    k_neighbors  = 15,
    n_iterations = 1,
    self_weight  = 0.5,
    max_radius   = None,
):
    """
    Apply spatial graph convolution to morphology features.

    For each cell i at each iteration:
        X_new[i] = self_weight * X[i]
                 + (1-self_weight) * inv_dist_weighted_mean(X[neighbors])

    Features are L2-normalized after every iteration so dot-product
    similarity remains in [-1, 1] for match_combined.

    Parameters
    ----------
    xy          : (n, 2)  spatial coordinates in µm
    features    : (n, d)  L2-normalized feature matrix (e.g. PCA output)
    k_neighbors : int     spatial kNN for graph construction
    n_iterations: int     number of convolution steps
                          1 → 1-hop neighbourhood
                          2 → 2-hop neighbourhood (receptive field doubles)
    self_weight : float   fraction of own feature retained each step
                          0.5 = equal self/neighbour (recommended start)
                          0.7 = stronger individual identity
                          0.3 = stronger neighbourhood context
    max_radius  : float|None  if set, ignore neighbours further than this (µm)
                              None = use all k_neighbors regardless of distance

    Returns
    -------
    X_smooth : (n, d) float32, L2-normalized
    """
    n    = len(xy)
    tree = cKDTree(xy)

    k_query = min(k_neighbors + 1, n)          # +1 because query returns self
    nb_dist, nb_idx = tree.query(xy, k=k_query)

    # Drop self (always index 0 at distance 0)
    nb_idx  = nb_idx[:, 1:]    # (n, k_neighbors)
    nb_dist = nb_dist[:, 1:]   # (n, k_neighbors)

    # ── Optional radius mask ──────────────────────────────────────────────────
    if max_radius is not None:
        valid = nb_dist <= max_radius                         # (n, k)
        # Replace masked neighbours with the cell itself to keep array shape;
        # their weight will be zero so they contribute nothing.
        nb_idx_use  = np.where(valid, nb_idx,
                               np.arange(n)[:, None])
        nb_dist_use = np.where(valid, nb_dist, 1e-6)
    else:
        valid       = np.ones(nb_dist.shape, dtype=bool)
        nb_idx_use  = nb_idx
        nb_dist_use = nb_dist

    # ── Inverse-distance weights, row-normalised ──────────────────────────────
    weights = 1.0 / (nb_dist_use + 1e-6)   # (n, k)
    weights = weights * valid               # zero out masked neighbours
    row_sum = weights.sum(axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, 1e-8)     # avoid 0/0 for isolated cells
    weights = weights / row_sum             # (n, k) sums to 1 per row

    # ── Iterative smoothing ───────────────────────────────────────────────────
    X = normalize(features.astype(np.float64))   # ensure L2-normalised input

    for step in range(n_iterations):
        nb_feats = X[nb_idx_use]                          # (n, k, d)
        nb_mean  = (weights[:, :, None] * nb_feats).sum(axis=1)  # (n, d)
        X_new    = self_weight * X + (1.0 - self_weight) * nb_mean
        X        = normalize(X_new)                        # re-normalise

    return X.astype(np.float32)





# ─────────────────────────────────────────────────────────────────────────────
#  Modified feature preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_morphology_features_graph(
    uni_panel1_list,
    uni_panel2_list,
    xy_panel1_list,          # NEW: spatial coords for each panel1 sample
    xy_panel2_list,          # NEW: spatial coords for each panel2 sample
    N_PCA_MORPH  = 512,
    k_neighbors  = 15,
    n_iterations = 1,
    self_weight  = 0.5,
    max_radius   = None,
    verbose      = True,
):
    """
    PCA-compress UNI features jointly, then apply spatial graph smoothing
    INDEPENDENTLY on each panel using its own spatial coordinates.

    Independence is critical: we never use registered positions or assume
    any spatial correspondence between panels — that would be circular.
    """
    if verbose:
        print("=" * 55)
        print("  Preparing graph-smoothed morphology features")
        print("=" * 55)

    # ── Step 1: Joint PCA (identical to prepare_morphology_features) ──────────
    all_X = []
    for adata in uni_panel1_list + uni_panel2_list:
        X = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)
        all_X.append(X.astype(np.float32))
    all_X = np.vstack(all_X)

    n_comps = min(N_PCA_MORPH, all_X.shape[1] - 1)
    if verbose:
        print(f"  Fitting PCA on {all_X.shape} → {n_comps} components...")

    pca = PCA(n_components=n_comps, random_state=42)
    pca.fit(all_X)

    if verbose:
        print(f"  Variance retained: {pca.explained_variance_ratio_.sum():.1%}")
    del all_X

    morph_panel1_list = []
    morph_panel2_list = []

    for uni_p1, uni_p2, xy_p1, xy_p2 in zip(
        uni_panel1_list, uni_panel2_list,
        xy_panel1_list,  xy_panel2_list,
    ):
        X_p1 = uni_p1.X.toarray() if issparse(uni_p1.X) else np.array(uni_p1.X)
        X_p2 = uni_p2.X.toarray() if issparse(uni_p2.X) else np.array(uni_p2.X)

        # PCA + L2 normalise (raw features before graph smoothing)
        m_p1_raw = normalize(pca.transform(X_p1.astype(np.float32)))
        m_p2_raw = normalize(pca.transform(X_p2.astype(np.float32)))

        if verbose:
            print(f"\n  Graph smoothing  (k={k_neighbors}, "
                  f"iters={n_iterations}, self_weight={self_weight})")

        # ── Step 2: Graph smooth each panel on its OWN coordinates ────────────
        m_p1 = graph_smooth_features(
            xy_p1, m_p1_raw,
            k_neighbors  = k_neighbors,
            n_iterations = n_iterations,
            self_weight  = self_weight,
            max_radius   = max_radius,
        )
        m_p2 = graph_smooth_features(
            xy_p2, m_p2_raw,
            k_neighbors  = k_neighbors,
            n_iterations = n_iterations,
            self_weight  = self_weight,
            max_radius   = max_radius,
        )

        if verbose:
            # How much did smoothing actually change the features?
            delta_p1 = float(np.linalg.norm(m_p1 - m_p1_raw, axis=1).mean())
            delta_p2 = float(np.linalg.norm(m_p2 - m_p2_raw, axis=1).mean())
            print(f"  Mean Δfeature from smoothing: "
                  f"P1={delta_p1:.4f}  P2={delta_p2:.4f}")
            print(f"  (0=no change, √2≈1.41=maximum change)")
            print(f"  Panel 1 morph {m_p1.shape}, "
                  f"Panel 2 morph {m_p2.shape}")

        morph_panel1_list.append(m_p1)
        morph_panel2_list.append(m_p2)

    return morph_panel1_list, morph_panel2_list, pca