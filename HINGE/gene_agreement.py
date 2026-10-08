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