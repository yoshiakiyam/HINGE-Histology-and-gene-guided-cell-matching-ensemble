
import numpy as np

from scipy.sparse import issparse
from sklearn.preprocessing import normalize
import scanpy as sc
from scipy.spatial import cKDTree


def _to_dense(X):
    if issparse(X):
        return np.asarray(X.toarray(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


def _expr_matrix(adata, genes):
    return _to_dense(adata[:, genes].X)


def _uni_matrix(uni):
    if hasattr(uni, "X"):
        return _to_dense(uni.X)
    return np.asarray(uni, dtype=np.float64)


def _xy(adata, spatial_key="spatial"):
    if spatial_key in adata.obsm:
        return np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    return np.asarray(adata.obsm["spatial"], dtype=np.float64)


def _mean_and_std(arr, axis=0):
    return arr.mean(axis=axis), arr.std(axis=axis)



#return the PCA-transformed expression values for the given genes in the adata object
def get_expr_pca(adata, genes, pca, NORMALIZE_EXPR=False):
    
    if NORMALIZE_EXPR:
        a = adata[:, genes].copy()
        sc.pp.normalize_total(a)
        sc.pp.log1p(a)
        X = _to_dense(a.X)
    else:
        X = _expr_matrix(adata, genes)
    return normalize(pca.transform(X))


def calculate_nn_dist(xy_p1, xy_p2
    ):

    
    
    # ── Coordinates: empirical displacement field ─────────────────
    tree = cKDTree(xy_p2)
    #get the closest point of p2 for every point in p1, 
    # then calculate the distance for each point
    nn_dist, _ = tree.query(xy_p1)
    
    return nn_dist


def compute_population_diff_stats(
    adata_panel1,
    uni_panel1_normed,
    adata_panel2,
    uni_panel2_normed,
    overlap_genes,
    spatial_key_p1="spatial",
    spatial_key_p2="spatial_registered",
    max_match_dist=None,
):
    """
    Compute population-level difference statistics + empirical spatial displacement field.

    uni_panel1_normed / uni_panel2_normed:
        Pass morph_panel1_list[i] / morph_panel2_list[i] directly.
        These are already L2-normalized.
    """
    print("=" * 55)
    print("  Computing population-level difference statistics")
    print(f"  Panel1: {len(adata_panel1)} cells  |  "
          f"Panel2: {len(adata_panel2)} cells")
    print("=" * 55)

    xy_p1 = _xy(adata_panel1, spatial_key_p1)
    xy_p2 = _xy(adata_panel2, spatial_key_p2)

    uni_p1 = np.asarray(uni_panel1_normed.X, dtype=np.float64)
    uni_p2 = np.asarray(uni_panel2_normed.X, dtype=np.float64)

    expr_p1 = _expr_matrix(adata_panel1, overlap_genes)
    expr_p2 = _expr_matrix(adata_panel2, overlap_genes)

    # ── Coordinates: empirical displacement field ─────────────────
    tree = cKDTree(xy_p2)
    #get the closest point of p2 for every point in p1, 
    # then calculate the distance for each point
    nn_dist, nn_idx = tree.query(xy_p1)
    nn_disp = xy_p2[nn_idx] - xy_p1

    #if too many cells are filterd out due to max_match_dist, keep all of them
    if max_match_dist is not None:
        keep_mask = nn_dist <= max_match_dist
        if keep_mask.sum() < 10:
            print(f"  Warning: only {keep_mask.sum()} matches within {max_match_dist} um; "
                  f"falling back to all nearest neighbors.")
            keep_mask = np.ones(len(xy_p1), dtype=bool)

    else:
        keep_mask = np.ones(len(xy_p1), dtype=bool)

    # robust summary
    delta_mean_x = float(np.median(nn_disp[keep_mask, 0]))
    delta_mean_y = float(np.median(nn_disp[keep_mask, 1]))
    coord_noise_std = float(np.median(nn_dist[keep_mask]))

    # ── UNI features ───────────────────────────────────────────────
    mean_uni_p1 = uni_p1.mean(axis=0)
    mean_uni_p2 = uni_p2.mean(axis=0)
    delta_mean_uni = mean_uni_p2 - mean_uni_p1

    std_uni_p1 = uni_p1.std(axis=0)
    std_uni_p2 = uni_p2.std(axis=0)
    delta_std_uni = np.sqrt((std_uni_p1**2 + std_uni_p2**2) / 2.0)

    # ── Expression ─────────────────────────────────────────────────
    mean_expr_p1 = expr_p1.mean(axis=0)
    mean_expr_p2 = expr_p2.mean(axis=0)
    delta_mean_expr = mean_expr_p2 - mean_expr_p1

    std_expr_p1 = expr_p1.std(axis=0)
    std_expr_p2 = expr_p2.std(axis=0)
    delta_std_expr = np.sqrt((std_expr_p1**2 + std_expr_p2**2) / 2.0)

    stats = {
        'delta_mean_x': delta_mean_x,
        'delta_mean_y': delta_mean_y,
        'coord_noise_std': coord_noise_std,

        'delta_mean_uni': delta_mean_uni,
        'delta_std_uni': np.maximum(delta_std_uni, 1e-8),

        'delta_mean_expr': delta_mean_expr,
        'delta_std_expr': np.maximum(delta_std_expr, 1e-8),

        'overlap_genes': list(overlap_genes),
        'n_panel1': len(adata_panel1),
        'n_panel2': len(adata_panel2),

        # NEW: empirical spatial displacement field
        'xy_anchor': xy_p1.copy(),
        'xy_nn_disp': nn_disp.copy(),
        'xy_nn_dist': nn_dist.copy(),
        'xy_keep_mask': keep_mask.copy(),
        'max_match_dist': max_match_dist,
    }

    print(f"\n  Coordinate displacement field:")
    print(f"    median dx        = {delta_mean_x:+.2f} um")
    print(f"    median dy        = {delta_mean_y:+.2f} um")
    print(f"    median NN dist   = {coord_noise_std:.2f} um")
    print(f"    kept fraction    = {keep_mask.mean():.1%}")

    print(f"\n  UNI (normalized space, {uni_p1.shape[1]}D):")
    print(f"    |delta_mean| median = {np.median(np.abs(delta_mean_uni)):.5f}")
    print(f"    delta_std    median = {np.median(delta_std_uni):.5f}")

    print(f"\n  Expression ({len(overlap_genes)} genes):")
    print(f"    |delta_mean| median = {np.median(np.abs(delta_mean_expr)):.4f}")
    print(f"    delta_std    median = {np.median(delta_std_expr):.4f}")

    return stats