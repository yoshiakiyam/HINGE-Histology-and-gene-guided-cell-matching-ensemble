import numpy as np
from HINGE.utils import _xy, _uni_matrix, _expr_matrix, _to_dense
from sklearn.preprocessing import normalize

import anndata as ad


def compute_population_diff_stats(
    adata_panel1,
    uni_panel1_normed,      
    adata_panel2,
    uni_panel2_normed,     
    overlap_genes,
    spatial_key_p1 = "spatial",
    spatial_key_p2 = "spatial_registered",
):
    """
    Compute population-level difference statistics.

    uni_panel1_normed / uni_panel2_normed:
        Pass morph_panel1_list[i] / morph_panel2_list[i] directly.
        These are already L2-normalized — consistent with what
        the matching functions use for cosine similarity.
    """
    print("=" * 55)
    print("  Computing population-level difference statistics")
    print(f"  Panel1: {len(adata_panel1)} cells  |  "
          f"Panel2: {len(adata_panel2)} cells")
    print("=" * 55)

    xy_p1   = _xy(adata_panel1, spatial_key_p1)
    xy_p2   = _xy(adata_panel2, spatial_key_p2)

    # Use normalized UNI directly 
    uni_p1  = np.asarray(uni_panel1_normed, dtype=np.float64)
    uni_p2  = np.asarray(uni_panel2_normed, dtype=np.float64)

    expr_p1 = _expr_matrix(adata_panel1, overlap_genes)
    expr_p2 = _expr_matrix(adata_panel2, overlap_genes)

    # ── Coordinates ───────────────────────────────────────────────
    delta_mean_x = float(xy_p2[:, 0].mean() - xy_p1[:, 0].mean())
    delta_mean_y = float(xy_p2[:, 1].mean() - xy_p1[:, 1].mean())
    coord_noise_std = float(
        0.02 * np.sqrt((xy_p1.std(axis=0)**2 +
                        xy_p2.std(axis=0)**2).mean() / 2)
    )

    # ── UNI features (normalized space) ──────────────────────────
    mean_uni_p1 = uni_p1.mean(axis=0)
    mean_uni_p2 = uni_p2.mean(axis=0)
    delta_mean_uni = mean_uni_p2 - mean_uni_p1

    std_uni_p1 = uni_p1.std(axis=0)
    std_uni_p2 = uni_p2.std(axis=0)
    delta_std_uni = np.sqrt((std_uni_p1**2 + std_uni_p2**2) / 2.0)

    # ── Expression ────────────────────────────────────────────────
    mean_expr_p1 = expr_p1.mean(axis=0)
    mean_expr_p2 = expr_p2.mean(axis=0)
    delta_mean_expr = mean_expr_p2 - mean_expr_p1

    std_expr_p1 = expr_p1.std(axis=0)
    std_expr_p2 = expr_p2.std(axis=0)
    delta_std_expr = np.sqrt((std_expr_p1**2 + std_expr_p2**2) / 2.0)

    stats = {
        'delta_mean_x'    : delta_mean_x,
        'delta_mean_y'    : delta_mean_y,
        'coord_noise_std' : coord_noise_std,
        'delta_mean_uni'  : delta_mean_uni,
        'delta_std_uni'   : np.maximum(delta_std_uni, 1e-8),
        'delta_mean_expr' : delta_mean_expr,
        'delta_std_expr'  : np.maximum(delta_std_expr, 1e-8),
        'overlap_genes'   : list(overlap_genes),
        'n_panel1'        : len(adata_panel1),
        'n_panel2'        : len(adata_panel2),
    }

    print(f"\n  Coordinate shift:")
    print(f"    delta_mean_x    = {delta_mean_x:+.2f} um")
    print(f"    delta_mean_y    = {delta_mean_y:+.2f} um")
    print(f"    coord_noise_std = {coord_noise_std:.4f} um")
    print(f"\n  UNI (normalized space, {uni_p1.shape[1]}D):")
    print(f"    |delta_mean| median = {np.median(np.abs(delta_mean_uni)):.5f}")
    print(f"    delta_std    median = {np.median(delta_std_uni):.5f}")
    print(f"\n  Expression ({len(overlap_genes)} genes):")
    print(f"    |delta_mean| median = {np.median(np.abs(delta_mean_expr)):.4f}")
    print(f"    delta_std    median = {np.median(delta_std_expr):.4f}")

    return stats



#returns a synthetic sample with scattered xy, uni, overlapping gene expression
def create_synthetic_sample(
    adata_panel1,
    uni_panel1_normed,      
    stats,
    alpha       = 1.0,
    noise_scale = 1.0,
    random_seed = 0,
    spatial_key = "spatial",
    RECENTER    = True,
):
    """
    Create synthetic s1_t using normalized UNI features.

    uni_panel1_normed: pass morph_panel1_list[i] directly.

    All arithmetic is done in normalized space.
    After adding noise, we re-normalize — consistent with
    what the matching functions receive.
    """
    rng = np.random.default_rng(random_seed)
    n1  = len(adata_panel1)

    xy_p1   = _xy(adata_panel1, spatial_key)
    uni_p1  = np.asarray(uni_panel1_normed.X, dtype=np.float64)  # normalized
    genes   = stats['overlap_genes']
    expr_p1 = _expr_matrix(adata_panel1, genes)

    # ── Coordinates ───────────────────────────────────────────────
    coord_noise = rng.normal(0, noise_scale * stats['coord_noise_std'],
                              size=(n1, 2))
    xy_t = xy_p1.copy()
    xy_t[:, 0] += alpha * stats['delta_mean_x'] + coord_noise[:, 0]
    xy_t[:, 1] += alpha * stats['delta_mean_y'] + coord_noise[:, 1]

    # ── UNI features (normalized space) ──────────────────────────
    # Add shift + noise in normalized space, then re-normalize
    # → stays on the unit hypersphere, consistent with morph_panel1_list
    uni_noise = rng.normal(
        0,
        noise_scale * stats['delta_std_uni'],
        size=uni_p1.shape
    )
    uni_t = uni_p1 + alpha * stats['delta_mean_uni'] + uni_noise
    uni_t = normalize(uni_t.astype(np.float32))    # re-normalize


    #create the anndata object
    uni_t_adata = uni_panel1_normed.copy()
    
    uni_t_adata.X = uni_t
    uni_t_adata.obsm["spatial"] = xy_t.copy()

    # ── Expression ────────────────────────────────────────────────
    expr_noise = rng.normal(
        0,
        noise_scale * stats['delta_std_expr'],
        size=expr_p1.shape
    )
    expr_t = np.clip(
        expr_p1 + alpha * stats['delta_mean_expr'] + expr_noise,
        0, None
    )

    # ── RECENTER ──────────────────────────────────────────────────
    xy_raw = xy_t.copy()
    if RECENTER:
        mean_shift = xy_t.mean(axis=0) - xy_p1.mean(axis=0)
        xy_t = xy_t - mean_shift

    # ── AnnData ───────────────────────────────────────────────────
    # create adata with blured gene expression for the overlapping genes
    adata_t = adata_panel1.copy()
    adata_t.obsm['spatial'] = xy_t.copy()

    X_t   = _to_dense(adata_t.X)
    gene_idx = adata_panel1.var_names.get_indexer(genes)
    if (gene_idx < 0).any():
        missing = [g for g, idx in zip(genes, gene_idx) if idx < 0]
        raise KeyError(f"genes missing from panel1: {missing[:5]}")
    X_t[:, gene_idx] = expr_t 
    adata_t.X = X_t

    coord_shift = np.linalg.norm(xy_t - xy_p1, axis=1).mean()
    print(f"  Created s1_t: alpha={alpha:.2f}  seed={random_seed}  "
          f"mean coord shift={coord_shift:.2f} um  n={n1}")

    return {
        'xy'           : xy_t,
        'xy_raw'       : xy_raw,
        'uni'          : uni_t_adata,
        'adata'        : adata_t,
        'cell_ids'     : np.array(adata_panel1.obs_names),
        'ground_truth' : np.arange(n1),
        'alpha'        : alpha,
        'random_seed'  : random_seed,
    }
    
    
#create multiple synthetic data
def create_synthetic_replicates(
    adata_panel1,
    uni_panel1_normed,
    stats,
    alpha_list        = [0.25, 0.5, 0.75, 1.0],
    n_seeds_per_alpha = 3,
    noise_scale       = 1.0,
    seed0             = 0,
    spatial_key       = "spatial",
    RECENTER          = True,
):
    """
    Create multiple synthetic replicates sweeping alpha and random seeds.
    """
    fakes = []
    rep   = 0

    for alpha in alpha_list:
        for seed_offset in range(n_seeds_per_alpha):
            seed = seed0 + rep
            fake = create_synthetic_sample(
                adata_panel1 = adata_panel1,
                uni_panel1_normed   = uni_panel1_normed,
                stats        = stats,
                alpha        = alpha,
                noise_scale  = noise_scale,
                random_seed  = seed,
                spatial_key  = spatial_key,
                RECENTER     = RECENTER,
            )
            fake['replicate'] = rep
            fakes.append(fake)
            rep += 1

    print(f"\n  Total replicates: {len(fakes)}")
    return fakes





