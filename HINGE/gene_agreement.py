import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import scanpy as sc


from HINGE.utils import _expr_matrix, _get_X







#returns a df with gene statistics, used later for selecting genes and calculating weights
def compute_gene_agreement_population(
    adata_panel1,
    adata_panel2,
    overlap_genes,
    min_detection_rate = 0.05,
    max_log2fc         = 1.0,
    verbose            = True,
):
    """
    Compute cross-panel gene agreement using POPULATION-LEVEL statistics.

    For each overlapping gene:
        mean_p1, mean_p2     = mean expression across all cells in each panel
        detection_p1/p2      = fraction of cells expressing the gene
        log2fc               = log2(mean_p2 / mean_p1)  — fold change
        cv_p1, cv_p2         = coefficient of variation (signal variability)
        agreement_score      = composite score in [0,1]

    No spatial pairing is used. This avoids any bias toward
    spatially-correlated or distance-matched genes.

    A gene agrees between panels if:
        1. It is detected in both panels (min_detection_rate)
        2. Its mean expression is similar (|log2fc| < max_log2fc)
        3. It shows variability in both panels (cv > 0)
           — flat/zero genes are uninformative for matching

    Parameters
    ----------
    max_log2fc : float
        Maximum allowed fold change between panel means.
        1.0 → allow up to 2-fold difference  (recommended)
        0.5 → stricter, within 1.4-fold
        2.0 → lenient, up to 4-fold

    Returns
    -------
    gene_stats : pd.DataFrame
        index = gene name, with columns:
            mean_p1, mean_p2
            detection_p1, detection_p2
            log2fc           : log2(mean_p2/mean_p1), signed
            abs_log2fc       : |log2fc|, lower = more similar
            cv_p1, cv_p2     : coefficient of variation
            cv_min           : min(cv_p1, cv_p2), proxy for informativeness
            agreement_score  : composite [0,1]
            use_for_matching : bool
    """

    X_p1 = _expr_matrix(adata_panel1, overlap_genes)   # (n1, n_genes)
    X_p2 = _expr_matrix(adata_panel2, overlap_genes)   # (n2, n_genes)

    # ── Population-level statistics ───────────────────────────────
    mean_p1 = X_p1.mean(axis=0)                  # (n_genes,)
    mean_p2 = X_p2.mean(axis=0)
    std_p1  = X_p1.std(axis=0)
    std_p2  = X_p2.std(axis=0)
    
    #detection rate, fraction of cells expressing the gene
    det_p1  = (X_p1 > 0).mean(axis=0)
    det_p2  = (X_p2 > 0).mean(axis=0)

    # Coefficient of variation (how variable the gene is)
    cv_p1 = std_p1 / (mean_p1 + 1e-8)
    cv_p2 = std_p2 / (mean_p2 + 1e-8)

    # Log2 fold change (signed)
    # if mean_p1 and mean_p2 is similar, log2fc will be close to 0
    # if the means are very different, log2fc will be large in magnitude
    log2fc = np.log2((mean_p2 + 1e-6) / (mean_p1 + 1e-6))

    # ── Build DataFrame ───────────────────────────────────────────
    df = pd.DataFrame({
        'mean_p1'     : mean_p1,
        'mean_p2'     : mean_p2,
        'detection_p1': det_p1,
        'detection_p2': det_p2,
        'log2fc'      : log2fc,
        'abs_log2fc'  : np.abs(log2fc),
        'cv_p1'       : cv_p1,
        'cv_p2'       : cv_p2,
        'cv_min'      : np.minimum(cv_p1, cv_p2),
    }, index=overlap_genes)

    # ── Quality filters ───────────────────────────────────────────
    passes_detection = (
        (df['detection_p1'] >= min_detection_rate) &
        (df['detection_p2'] >= min_detection_rate)
    )
    passes_fc = df['abs_log2fc'] <= max_log2fc
    passes_cv = df['cv_min']    >  0.01   # must show some variability

    #passes if all three is true
    df['use_for_matching'] = passes_detection & passes_fc & passes_cv

    # ── Agreement score ───────────────────────────────────────────
    # Three components, each in [0, 1]:
    #
    # 1. FC score: how similar are the means?
    #    abs_log2fc=0   → fc_score=1.0  (identical means)
    #    abs_log2fc=1   → fc_score=0.5  (2-fold difference)
    #    abs_log2fc=2   → fc_score=0.0  (4-fold difference, clipped to 0)
    fc_score = np.clip(1.0 - df['abs_log2fc'] / 2.0, 0, 1)

    # 2. Detection score: gene must be detected in both panels
    det_score = np.sqrt(df['detection_p1'] * df['detection_p2'])

    # 3. Variability score: uninformative (near-zero) genes penalized
    #    cv_min > 0.5 → fully informative
    #    cv_min = 0   → uninformative
    cv_score = np.clip(df['cv_min'] / 0.5, 0, 1)

    df['fc_score']   = fc_score
    df['det_score']  = det_score
    df['cv_score']   = cv_score
    
    # agreement_score - max 1, min 0
    df['agreement_score'] = fc_score * det_score * cv_score

    # Zero out excluded genes
    df.loc[~df['use_for_matching'], 'agreement_score'] = 0.0

    # print stats if needed
    if verbose:
        n_tot  = len(df)
        n_use  = df['use_for_matching'].sum()
        n_fc   = passes_fc.sum()
        n_det  = passes_detection.sum()

        print(f"\n  Population-level gene agreement:")
        print(f"    Total overlapping genes      : {n_tot}")
        print(f"    Pass detection (≥{min_detection_rate:.0%}): "
              f"{n_det}  ({n_det/n_tot:.1%})")
        print(f"    Pass fold change (≤{max_log2fc:.1f} log2): "
              f"{n_fc}   ({n_fc/n_tot:.1%})")
        print(f"    Pass all filters             : "
              f"{n_use}  ({n_use/n_tot:.1%})")
        print(f"\n    Agreement score (passing genes):")
        passing = df.loc[df['use_for_matching'], 'agreement_score']
        print(f"      median : {passing.median():.3f}")
        print(f"      mean   : {passing.mean():.3f}")
        print(f"      top 10 genes:")
        top10 = df.nlargest(10, 'agreement_score')[
            ['mean_p1', 'mean_p2', 'log2fc',
             'detection_p1', 'detection_p2', 'agreement_score']
        ]
        print(top10.to_string())

    return df






def build_weighted_expression_features(
    adata_panel1,
    adata_panel2,
    gene_stats,
    min_r          = 0.0,      # min agreement_score to include in the gene to use
    top_n_genes    = None,
    normalize_expr = False,
    verbose        = True,
):
    """
    Build expression feature vectors using only population-agreed genes,
    weighted by their agreement score.

    Returns
    -------
    expr_p1  : (n1, N_PCA_MORPH) L2-normalized
    expr_p2  : (n2, N_PCA_MORPH) L2-normalized
    selected : list of str
    w_expr   : float in [0, 0.5]
        Adaptive weight for expression in combined matching.
        Derived from overall quality of gene agreement.
    """
    

    # ── Select genes that pass filters──────────────────────────────────────────────
    selected_df = gene_stats[
        gene_stats['use_for_matching'] &
        (gene_stats['agreement_score'] >= min_r)
    ].copy()

    #select top_n genes if set
    if top_n_genes is not None:
        selected_df = selected_df.nlargest(top_n_genes, 'agreement_score')

    selected_genes = selected_df.index.tolist()
    n_selected     = len(selected_genes)
    n_total        = len(gene_stats)

    if n_selected == 0:
        print("  ⚠️  No genes passed selection → expression weight = 0.0")
        return (np.zeros((len(adata_panel1), 1)),
                np.zeros((len(adata_panel2), 1)),
                [], 0.0)



    # ── Compute adaptive expression weight ───────────────────────
    # Based purely on population agreement — no spatial information
    #
    # Quality signal = (fraction of genes selected) × (mean agreement score)
    #   → 0.0 if no genes agree
    #   → 0.5 if half the genes agree well (score ≥ 0.5)
    frac_selected    = n_selected / n_total
    mean_agreement   = float(selected_df['agreement_score'].mean())
    quality_signal   = frac_selected * mean_agreement

    #calculate feature weight based on the quality of the expression
    # Map to [0, 0.5]: saturates at 0.5 when quality_signal ≥ 0.25
    w_expr = float(np.clip( 0.5 * quality_signal, 0.0, 0.5))
    w_morph = 1.0 - w_expr

    if verbose:
        print(f"\n  Expression feature construction:")
        print(f"    Selected genes    : {n_selected} / {n_total}  "
              f"({n_selected/n_total:.1%})")
        print(f"    Mean agreement    : {mean_agreement:.3f}")
        print(f"    Frac selected     : {frac_selected:.3f}")
        print(f"    Quality signal    : {quality_signal:.3f}")
        print(f"    → w_expr  = {w_expr:.3f}")
        print(f"    → w_morph = {w_morph:.3f}")








    # ── Get expression matrices ───────────────────────────────────


    X_p1 = _get_X(adata_panel1, selected_genes, normalize_expr)
    X_p2 = _get_X(adata_panel2, selected_genes, normalize_expr)

    # ── Weight genes by agreement score ──────────────────────────
    weights = selected_df.loc[selected_genes, 'agreement_score'].values
    weights = weights / (weights.sum() + 1e-8)

    X_p1 = X_p1 * weights[None, :]
    X_p2 = X_p2 * weights[None, :]


    return X_p1, X_p2, selected_genes, w_expr


def plot_gene_agreement(gene_stats, save_path=None):
    """
    Visualize population-level gene agreement between panels.
    """
    use   = gene_stats['use_for_matching']

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('Population-level Gene Agreement (480 vs 5k panels)',
                 fontsize=12, fontweight='bold')

    # ── Scatter: mean_p1 vs mean_p2 ──────────────────────────────
    ax = axes[0]
    ax.scatter(gene_stats.loc[~use, 'mean_p1'],
               gene_stats.loc[~use, 'mean_p2'],
               s=8, alpha=0.4, c='#e74c3c', label='excluded')
    ax.scatter(gene_stats.loc[use, 'mean_p1'],
               gene_stats.loc[use,  'mean_p2'],
               s=8, alpha=0.6,
               c=gene_stats.loc[use, 'agreement_score'],
               cmap='RdYlGn', vmin=0, vmax=1,
               label='selected')

    lim = max(gene_stats['mean_p1'].max(), gene_stats['mean_p2'].max())
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1, alpha=0.5,
            label='y=x')
    ax.set_xlabel('Mean expression — Panel1 (480)')
    ax.set_ylabel('Mean expression — Panel2 (5k)')
    ax.set_title('Mean expression comparison\n(color=agreement score)')
    ax.legend(fontsize=8)

    # ── Log2 FC distribution ──────────────────────────────────────
    ax = axes[1]
    ax.hist(gene_stats.loc[~use, 'log2fc'],
            bins=50, color='#e74c3c', alpha=0.6, label='excluded')
    ax.hist(gene_stats.loc[use,  'log2fc'],
            bins=50, color='#2ecc71', alpha=0.6, label='selected')
    ax.axvline( 1.0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(-1.0, color='gray', linestyle='--', linewidth=1,
               label='±1 log2FC cutoff')
    ax.axvline(0,    color='black', linestyle='-', linewidth=1,
               alpha=0.5, label='no change')
    ax.set_xlabel('log2FC (panel2 / panel1)')
    ax.set_ylabel('Number of genes')
    ax.set_title('Fold change distribution\n(|log2FC| ≤ 1.0 = selected)')
    ax.legend(fontsize=8)

    # ── Agreement score distribution ─────────────────────────────
    ax = axes[2]
    ax.hist(gene_stats.loc[use, 'agreement_score'],
            bins=30, color='steelblue', alpha=0.8)
    ax.axvline(gene_stats.loc[use, 'agreement_score'].median(),
               color='red', linestyle='--',
               label=f"median = "
                     f"{gene_stats.loc[use,'agreement_score'].median():.3f}")
    ax.set_xlabel('Agreement score')
    ax.set_ylabel('Number of genes')
    ax.set_title(f'Agreement score distribution\n'
                 f'(n={use.sum()} selected genes)')
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=130, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
#################################################################################################################################################################################
#clustering based

from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsClassifier



def _normalize_for_knn(X_raw):
    """Log1p + L2 normalize for cosine KNN. Same transform applied to both panels."""

    X = np.log1p(X_raw.astype(np.float64))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, 1e-8, None)

def compute_cluster_structure_score(
    adata_panel1,
    adata_panel2,
    overlap_genes,
    n_pca_comps           = 30,
    leiden_resolution     = 0.5,
    n_neighbors_transfer  = 15,    # KNN for label transfer
    min_cells_per_cluster = 20,
    min_valid_clusters    = 3,
    verbose               = True,
):
    """
    Cluster Panel 1 independently → transfer labels to Panel 2 via KNN
    on overlapping genes → compute pseudobulk Pearson r per gene.

    Why label transfer instead of joint clustering
    -----------------------------------------------
    Joint clustering of two panels with different gene counts (5101 vs 480)
    produces batch-separated clusters even after subsetting to overlapping
    genes, because per-gene expression distributions differ systematically.
    Label transfer avoids this: Panel 1 is clustered on its full gene set
    (richest biological signal), Panel 2 cells are then projected into
    that cluster space using only the shared genes.

    No spatial information is used at any step → no circularity.

    Parameters
    ----------
    n_neighbors_transfer : int
        Number of Panel 1 neighbors used for majority-vote label transfer.
    """
    
    

    # ── Raw expression on overlapping genes ──────────────────────────────────
    X_p1_overlap = _expr_matrix(adata_panel1, overlap_genes)  # (n1, G)
    X_p2_overlap = _expr_matrix(adata_panel2, overlap_genes)  # (n2, G)
    G = len(overlap_genes)# number of overlapping genes




    # ── Step 1: Cluster Panel 1 on its FULL gene set ─────────────────────────

    adata_p1 = adata_panel1.copy()
    sc.pp.normalize_total(adata_p1, target_sum=1e4)
    sc.pp.log1p(adata_p1)
    sc.pp.highly_variable_genes(adata_p1, n_top_genes=2000, flavor='seurat_v3')
    n_comps_safe = min(n_pca_comps, adata_p1.n_vars - 1, adata_p1.n_obs - 1)
    sc.pp.pca(adata_p1, n_comps=n_comps_safe, use_highly_variable=True)
    sc.pp.neighbors(adata_p1, n_pcs=n_comps_safe)
    sc.tl.leiden(adata_p1, resolution=leiden_resolution, key_added='cluster')

    labels_p1 = adata_p1.obs['cluster'].values.astype(str)
    unique_clusters = np.unique(labels_p1)
    K = len(unique_clusters) # number of clusters in panel 1

    if verbose:
        sizes = pd.Series(labels_p1).value_counts()
        print(f"\n  Panel 1 Leiden clustering  "
              f"(resolution={leiden_resolution}, full gene set)")
        print(f"    Clusters         : {K}")
        print(f"    Cluster sizes    : min={sizes.min()}, "
              f"median={int(sizes.median())}, max={sizes.max()}")




    # ── Step 2: Transfer labels to Panel 2 via KNN on overlapping genes ──────
    # Both panels are normalized the same way on the same gene set,
    # so KNN distances are meaningful.
    p1_norm = _normalize_for_knn(X_p1_overlap)   # log1p + L2 normalize
    p2_norm = _normalize_for_knn(X_p2_overlap)

    knn = KNeighborsClassifier(
        n_neighbors = min(n_neighbors_transfer, len(labels_p1)),
        metric      = 'cosine',
        n_jobs      = -1,
    )
    knn.fit(p1_norm, labels_p1) # fit KNN on panel 1
    labels_p2 = knn.predict(p2_norm)#transfer labels to panel 2

    if verbose:
        # Sanity check: label distribution should be similar in both panels
        # if the transfer worked (i.e., cell type composition is similar)
        dist_p1 = pd.Series(labels_p1).value_counts(normalize=True).sort_index()
        dist_p2 = pd.Series(labels_p2).value_counts(normalize=True).sort_index()
        print(f"\n  Label transfer to Panel 2  (KNN k={n_neighbors_transfer}, "
              f"metric=cosine)")
        print(f"    {'Cluster':<10} {'P1 frac':>10} {'P2 frac':>10}")
        for c in unique_clusters:
            f1 = dist_p1.get(c, 0.0)
            f2 = dist_p2.get(c, 0.0)
            flag = "  ⚠️ " if abs(f1 - f2) > 0.15 else ""
            print(f"    {c:<10} {f1:>10.3f} {f2:>10.3f}{flag}")
        print(f"    (large P1/P2 fraction differences may indicate "
              f"true composition differences between sections)")





    # ── Step 3: Pseudobulk per cluster, RAW expression ───────────────────────
    pb_p1 = np.zeros((K, G), dtype=np.float64)
    pb_p2 = np.zeros((K, G), dtype=np.float64)
    n_cells_p1 = np.zeros(K, dtype=int)
    n_cells_p2 = np.zeros(K, dtype=int)

    for k_idx, k in enumerate(unique_clusters):
        mask_p1 = labels_p1 == k
        mask_p2 = labels_p2 == k
        n_cells_p1[k_idx] = mask_p1.sum()
        n_cells_p2[k_idx] = mask_p2.sum()
        if mask_p1.sum() > 0:
            pb_p1[k_idx] = X_p1_overlap[mask_p1].mean(axis=0)
        if mask_p2.sum() > 0:
            pb_p2[k_idx] = X_p2_overlap[mask_p2].mean(axis=0)

    # Only use clusters with enough cells in BOTH panels
    valid = (n_cells_p1 >= min_cells_per_cluster) & \
            (n_cells_p2 >= min_cells_per_cluster)
    n_valid = int(valid.sum())

    if verbose:
        print(f"\n    Valid clusters   : {n_valid} / {K}  "
              f"(≥{min_cells_per_cluster} cells per panel)")

    if n_valid < min_valid_clusters:
        if verbose:
            print(f"  ⚠️  Only {n_valid} valid clusters — "
                  f"returning zero structure scores.  "
                  f"Try lower leiden_resolution.")
        return (pd.Series(np.zeros(G), index=overlap_genes),
                labels_p1, labels_p2)

    pb_p1_v = pb_p1[valid]   # (n_valid, G)
    pb_p2_v = pb_p2[valid]





    # ── Step 4: Per-gene Pearson r ────────────────────────────────────────────
    structure_scores = np.zeros(G)

    for g in range(G):
        v1 = pb_p1_v[:, g]
        v2 = pb_p2_v[:, g]

        # Flat in Panel 1 → gene has no discriminative pattern → skip
        # Flat in Panel 2 → gene not reliably measured there → skip
        if v1.std() < 1e-8 or v2.std() < 1e-8:
            structure_scores[g] = 0.0
            continue

        r, _ = pearsonr(v1, v2)
        structure_scores[g] = float(np.clip(
            0.0 if np.isnan(r) else r, 0.0, 1.0
        ))

    if verbose:
        informative = structure_scores > 0
        if informative.sum() > 0:
            sc_inf = structure_scores[informative]
            print(f"\n    Structure scores (r > 0): {informative.sum()} / {G} genes")
            print(f"      Median r : {np.median(sc_inf):.3f}")
            print(f"      Mean r   : {np.mean(sc_inf):.3f}")
            for t in [0.3, 0.5, 0.7, 0.9]:
                print(f"      r ≥ {t}  : {(sc_inf >= t).sum()}")

    return (pd.Series(structure_scores, index=overlap_genes),
            labels_p1, labels_p2)



def compute_gene_agreement_population_2(
    adata_panel1,
    adata_panel2,
    overlap_genes,
    min_detection_rate    = 0.02,   # relaxed from 0.05 — structure score
                                    # now protects against noisy rare genes
    max_log2fc            = 1.0,
    use_cluster_structure = True,   # new flag
    leiden_resolution     = 0.5,
    n_pca_comps           = 30,
    min_cells_per_cluster = 20,
    min_structure_score   = 0.0,    # hard filter: drop genes below this r
    verbose               = True,
):
    """
    Compute cross-panel gene agreement combining POPULATION-level statistics
    with CLUSTER-level structure preservation.

    Agreement score formula
    -----------------------
    Without cluster structure (use_cluster_structure=False):
        agreement_score = fc_score × det_score × cv_score      [original]

    With cluster structure (use_cluster_structure=True):
        agreement_score = fc_score × det_score × structure_score

        structure_score  (Pearson r of pseudobulk vectors, see
                          compute_cluster_structure_score) REPLACES cv_score
        because structure_score is strictly more informative:
          • flat genes already receive r = 0  (subsumes cv_score)
          • additionally penalizes genes whose cell-state pattern is not
            reproduced in Panel 2, catching technical/platform-specific genes

    Rare-gene handling
    ------------------
    min_detection_rate is relaxed to 0.02 when cluster structure is enabled.
    A gene expressed only in 2-3 % of cells (rare cell type marker) will
    form a small cluster in the joint Leiden embedding and receive a high
    structure_score if its pattern is preserved, so it will NOT be dropped.
    A gene with 2 % detection due to technical Xenium dropout will show
    inconsistent patterns → low structure_score → naturally down-weighted.

    Parameters
    ----------
    min_structure_score : float
        Hard minimum structure score to include a gene.  0.0 = no filter
        (soft weighting only).  Typical useful range: 0.0 – 0.3.

    Returns
    -------
    gene_stats : pd.DataFrame
        All original columns plus:
            structure_score  : Pearson r from pseudobulk comparison [0, 1]
            cluster_labels   : stored as metadata (not a column)
    """
    X_p1 = _expr_matrix(adata_panel1, overlap_genes)
    X_p2 = _expr_matrix(adata_panel2, overlap_genes)

    # ── Population-level statistics (unchanged) ───────────────────────────────
    mean_p1 = X_p1.mean(axis=0)
    mean_p2 = X_p2.mean(axis=0)
    std_p1  = X_p1.std(axis=0)
    std_p2  = X_p2.std(axis=0)
    det_p1  = (X_p1 > 0).mean(axis=0)
    det_p2  = (X_p2 > 0).mean(axis=0)
    cv_p1   = std_p1 / (mean_p1 + 1e-8)
    cv_p2   = std_p2 / (mean_p2 + 1e-8)
    log2fc  = np.log2((mean_p2 + 1e-6) / (mean_p1 + 1e-6))

    df = pd.DataFrame({
        'mean_p1'      : mean_p1,
        'mean_p2'      : mean_p2,
        'detection_p1' : det_p1,
        'detection_p2' : det_p2,
        'log2fc'       : log2fc,
        'abs_log2fc'   : np.abs(log2fc),
        'cv_p1'        : cv_p1,
        'cv_p2'        : cv_p2,
        'cv_min'       : np.minimum(cv_p1, cv_p2),
    }, index=overlap_genes)

    # ── Quality filters ───────────────────────────────────────────────────────
    passes_detection = (
        (df['detection_p1'] >= min_detection_rate) &
        (df['detection_p2'] >= min_detection_rate)
    )
    passes_fc = df['abs_log2fc'] <= max_log2fc
    passes_cv = df['cv_min']    >  0.01

    df['use_for_matching'] = passes_detection & passes_fc & passes_cv

    # ── Component scores ──────────────────────────────────────────────────────
    fc_score  = np.clip(1.0 - df['abs_log2fc'] / 2.0, 0, 1)
    det_score = np.sqrt(df['detection_p1'] * df['detection_p2'])
    cv_score  = np.clip(df['cv_min'] / 0.5, 0, 1)   # kept for diagnostics

    df['fc_score']  = fc_score
    df['det_score'] = det_score
    df['cv_score']  = cv_score

    # ── Cluster structure score ───────────────────────────────────────────────
    if use_cluster_structure:
        if verbose:
            print(f"\n=======================================================")
            print(f"  Computing cluster-level structure scores")
            print(f"=======================================================")

        structure_s, cluster_labels = compute_cluster_structure_score(
            adata_panel1          = adata_panel1,
            adata_panel2          = adata_panel2,
            overlap_genes         = overlap_genes,
            n_pca_comps           = n_pca_comps,
            leiden_resolution     = leiden_resolution,
            min_cells_per_cluster = min_cells_per_cluster,
            verbose               = verbose,
        )
        df['structure_score'] = structure_s.values

        # Optional hard filter
        if min_structure_score > 0.0:
            fails_structure = df['structure_score'] < min_structure_score
            df.loc[fails_structure, 'use_for_matching'] = False

        # Final agreement score: replaces cv_score with structure_score
        # Reasoning: structure_score captures variability (flat genes → r=0)
        # AND cross-panel reproducibility, which cv_score does not.
        df['agreement_score'] = fc_score * det_score * df['structure_score']

    else:
        # Original formula (fallback / ablation)
        df['structure_score'] = np.nan
        cluster_labels        = None
        df['agreement_score'] = fc_score * det_score * cv_score

    # Zero out genes that failed filters
    df.loc[~df['use_for_matching'], 'agreement_score'] = 0.0

    # ── Verbose summary ───────────────────────────────────────────────────────
    if verbose:
        n_tot = len(df)
        n_use = df['use_for_matching'].sum()
        n_fc  = passes_fc.sum()
        n_det = passes_detection.sum()

        print(f"\n  Population-level gene agreement:")
        print(f"    Total overlapping genes      : {n_tot}")
        print(f"    Pass detection (≥{min_detection_rate:.0%}): "
              f"{n_det}  ({n_det/n_tot:.1%})")
        print(f"    Pass fold change (≤{max_log2fc:.1f} log2): "
              f"{n_fc}   ({n_fc/n_tot:.1%})")
        print(f"    Pass all filters             : "
              f"{n_use}  ({n_use/n_tot:.1%})")

        passing = df.loc[df['use_for_matching'], 'agreement_score']
        print(f"\n    Agreement score (passing genes):")
        print(f"      median : {passing.median():.3f}")
        print(f"      mean   : {passing.mean():.3f}")
        print(f"      top 10 genes:")
        top10 = df.nlargest(10, 'agreement_score')[
            ['mean_p1', 'mean_p2', 'log2fc',
             'detection_p1', 'detection_p2',
             'structure_score', 'agreement_score']
        ]
        print(top10.to_string())

    # Attach cluster labels as metadata for external diagnostics
    df.attrs['cluster_labels'] = cluster_labels
    return df
