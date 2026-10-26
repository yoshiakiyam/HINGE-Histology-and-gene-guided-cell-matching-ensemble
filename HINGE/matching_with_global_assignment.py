
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
import anndata as ad

from HINGE.utils import calculate_nn_dist

def _global_assignment(
    xy_panel1,
    xy_panel2_reg,
    score_fn,
    max_dist=50.0,
    threshold=None,
    k_candidates=None,
):
    """
    Maximum-weight one-to-one matching over spatially valid pairs.

    Parameters
    ----------
    score_fn : callable
        score_fn(i, j) returns the score for pairing i with j.
        Higher scores are better.

    threshold : float or None
        Minimum acceptable score. Pairs below this are forbidden.
        None means no explicit threshold.

    k_candidates : int or None
        None: consider ALL panel2 points within max_dist.
        Integer: consider only the k nearest within max_dist.

    Returns
    -------
    matched_panel1, matched_panel2, matched_scores
    """
    n1 = len(xy_panel1)
    n2 = len(xy_panel2_reg)

    if n1 == 0 or n2 == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=float),
        )

    tree = cKDTree(xy_panel2_reg)

    # Find candidate panel2 points for each panel1 point.
    #if k_candidates is None, consider all within max_dist.
    if k_candidates is None:
        candidate_lists = tree.query_ball_point(
            xy_panel1, r=max_dist
        )
        
    #else, consider only the k nearest within max_dist.
    else:
        k = min(k_candidates, n2)
        dist, idx = tree.query(
            xy_panel1,
            k=k,
            distance_upper_bound=max_dist,
        )
        #convert to 2D arrays for consistent indexing, even if k=1.
        if k == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        # For each panel1 point, keep only valid candidates (idx < n2). 
        candidate_lists = [
            idx[i][idx[i] < n2].tolist()
            for i in range(n1)
        ]

    #a matrix with
    #n1 rows - one for each panel1 point
    #n2 + n1 columns - one for each panel2 point, plus n1 dummy unmatched columns
    #initially filled with a very negative score (-1e12) to represent forbidden pairs.
    scores = np.full((n1, n2 + n1), -1e12, dtype=float)

    # Dummy matches have score 0.
    # They represent leaving a cell unmatched
    # in case no valid candidate is available.
    scores[:, n2:] = 0.0


    # for every panel 1 point, compute the score for each of its candidate panel2 points.
    #skip any candidate whose score is below the threshold (if provided).
    for i, candidates in enumerate(candidate_lists):
        for j in candidates:
            score = float(score_fn(i, j))

            if threshold is not None and score < threshold:
                continue

            scores[i, j] = score
            
            
    # Solve the linear sum assignment problem to find the optimal one-to-one matching.
    row_ind, col_ind = linear_sum_assignment(
        scores, maximize=True
    )

    # Keep only real matches, excluding dummy columns.
    real = col_ind < n2
    matched_panel1 = row_ind[real]
    matched_panel2 = col_ind[real]

    matched_scores = scores[
        matched_panel1, matched_panel2
    ]

    # Remove any forbidden pairs, for safety.
    valid = matched_scores > -1e11

    return (
        matched_panel1[valid],
        matched_panel2[valid],
        matched_scores[valid],
    )

"""
Globally minimize total distance among valid matches.

Each valid pair gets score max_dist - distance.
This makes valid matches preferable to leaving cells unmatched,
while favoring shorter distances.
"""
def match_distance_only(
    xy_panel1,
    xy_panel2_reg,
    max_dist,
    k_candidates=None,
    verbose = False,
):


    #get euclidean distance between two points
    #higher score for shorter distance, with max score = max_dist
    def distance_score(i, j):
        d = np.linalg.norm(
            xy_panel1[i] - xy_panel2_reg[j]
        )
        return max_dist - d


    #get matched pairs using global assignment with distance scoring    
    matched_p1, matched_p2, _ = _global_assignment(
        xy_panel1,
        xy_panel2_reg,
        distance_score,
        max_dist=max_dist,
        k_candidates=k_candidates,
    )
    
    
    #compute distances for printing
    matched_dist = np.linalg.norm(
        xy_panel1[matched_p1] -
        xy_panel2_reg[matched_p2],
        axis=1,
    )

    if verbose:
        print(
            f"  [A distance   ] {len(matched_p1):>6} pairs "
            f"({len(matched_p1)/len(xy_panel1):.1%}), "
            f"median dist = {np.median(matched_dist):.2f}"
        )
    
    
    
    

    return matched_p1, matched_p2


"""
Match cells based on feature similarity using a feature 

"""
def match_feature_only(
    feat_panel1,
    feat_panel2,
    xy_panel1,
    xy_panel2_reg,
    max_dist,
    k_candidates=None,
    label="feature",
    threshold=None,
    verbose = False
):

    
    
    #function for computing feature similarity score between two points
    #by taking the dot product of their feature vectors
    def feature_score(i, j):
        return float(feat_panel1[i] @ feat_panel2[j])
    
    
    
    #get matched pairs using global assignment with feature scoring
    matched_p1, matched_p2, matched_sc = _global_assignment(
        xy_panel1,
        xy_panel2_reg,
        feature_score,
        max_dist=max_dist,
        threshold=threshold,
        k_candidates=k_candidates,
    )


    #compute distances for printing
    matched_dist = np.linalg.norm(
        xy_panel1[matched_p1] -
        xy_panel2_reg[matched_p2],
        axis=1,
    )

    if verbose:
        print(
            f"  [{label:13s}] {len(matched_p1):>6} pairs "
            f"({len(matched_p1)/len(xy_panel1):.1%}), "
            f"median dist = {np.median(matched_dist):.2f}, "
            f"median sim = {np.median(matched_sc):.3f}"
        )

    return matched_p1, matched_p2


"""
Match cells based on two feature similarities using a weighted combination of the two scores.

"""
def match_combined(
    feat1_panel1,
    feat1_panel2,
    feat2_panel1,
    feat2_panel2,
    xy_panel1,
    xy_panel2_reg,
    max_dist,
    w1=0.5,
    w2=0.5,
    k_candidates=None, # num of candidate matches to consider for global assignment
    threshold=None, # minimum similarity score for a match to be considered
    verbose = False
):
    #calculate combined score for two features using weights w1 and w2
    def combined_score(i, j):
        s1 = float(
            feat1_panel1[i] @ feat1_panel2[j]
        )
        s2 = float(
            feat2_panel1[i] @ feat2_panel2[j]
        )

        return w1 * s1 + w2 * s2

    #get matched pairs using global assignment with combined feature scoring
    matched_p1, matched_p2, matched_sc = _global_assignment(
        xy_panel1,
        xy_panel2_reg,
        combined_score,
        max_dist=max_dist,
        threshold=threshold,
        k_candidates=k_candidates,
    )


    #compute distances for printing
    matched_dist = np.linalg.norm(
        xy_panel1[matched_p1] -
        xy_panel2_reg[matched_p2],
        axis=1,
    )
    if verbose:
        print(
            f"  [D combined   ] {len(matched_p1):>6} pairs "
            f"({len(matched_p1)/len(xy_panel1):.1%}), "
            f"median dist = {np.median(matched_dist):.2f}, "
            f"median sim = {np.median(matched_sc):.3f}"
        )

    return matched_p1, matched_p2


    
    
##################################################################################

from HINGE.gene_agreement import compute_gene_agreement_population, build_weighted_expression_features
from HINGE.utils import get_overlapping_genes
from HINGE.feature_mapping import prepare_morphology_features, prepare_expression_features
def run_hinge(
    xy_panel1,
    xy_panel2_reg,
    morph_panel1,#NOT PCA COMPRESSED
    morph_panel2,#NOT PCA COMPRESSED
    expr_panel1,#NOT PCA COMPRESSED
    expr_panel2,#NOT PCA COMPRESSED
    MAX_DIST=None,
    k_candidates=None,
    threshold=None,
    N_PCA_MORPH = 256,
    n_gene_comps = 128,
    verbose = False
):
    if MAX_DIST is None:
        #automatically set distance
        
        
        nn_dist = calculate_nn_dist(xy_panel1, xy_panel2_reg)

        p50  = float(np.percentile(nn_dist, 50))
        p75  = float(np.percentile(nn_dist, 75))
        p90  = float(np.percentile(nn_dist, 90))
        p95  = float(np.percentile(nn_dist, 95))
        p99  = float(np.percentile(nn_dist, 99))

        if verbose:
            print("MAX_DIST is calculated automatically")
            print(f"  Core NN dist p50     : {p50:.2f} um")
            print(f"  Core NN dist p75     : {p75:.2f} um")
            print(f"  Core NN dist p90     : {p90:.2f} um")
            print(f"  Core NN dist p95     : {p95:.2f} um")
            print(f"  Core NN dist p99     : {p99:.2f} um")
            
            print(f"Using MAX DIST: {round(p90)}")

    
        MAX_DIST = round(p90)
        
        
        
    
    overlapping_genes = get_overlapping_genes([expr_panel1], [expr_panel2])
    if verbose:
        print(f"  Overlapping genes across all samples: {len(overlapping_genes)}")
    
    
    gene_stats = compute_gene_agreement_population(expr_panel1, expr_panel2, overlapping_genes,
                                                   min_detection_rate=0.05,
                                                   max_log2fc=1.0,
                                                   verbose = verbose)
    
    weighted_expr_p1, weighted_expr_p2, selected_genes, w_expr = build_weighted_expression_features(
        expr_panel1,
        expr_panel2,
        gene_stats,
        min_r          = 0.0,      # min agreement_score to include in the gene to use
        top_n_genes    = None,
        normalize_expr = False,
        verbose        = verbose,
    )
    
    
    # ── Prepare features ──────────────────────────────────────────────────────
    morph_p1_extracted_list, morph_pl2_extracted_list, morph_pca = prepare_morphology_features(
        [morph_panel1], [morph_panel2], N_PCA_MORPH=N_PCA_MORPH
    )

    #remake anndata for pca
    weighted_expr_p1_adata = ad.AnnData(X=weighted_expr_p1)
    weighted_expr_p1_adata.var_names = selected_genes
    
    weighted_expr_p2_adata = ad.AnnData(X=weighted_expr_p2)
    weighted_expr_p2_adata.var_names = selected_genes
    
    w_expr_extracted_p1_list, w_expr_extracted_p2_list, expr_pca, overlap_genes = prepare_expression_features(
    [weighted_expr_p1_adata],
    [weighted_expr_p2_adata],
    n_comps=n_gene_comps
    )
    
    
    return  match_combined(
                morph_p1_extracted_list[0],
                morph_pl2_extracted_list[0],
                w_expr_extracted_p1_list[0],
                w_expr_extracted_p2_list[0],
                xy_panel1,
                xy_panel2_reg,
                max_dist=MAX_DIST,
                w1=1-w_expr,
                w2=w_expr,
                k_candidates=k_candidates,
                threshold=threshold,
            )
    
####################################################################################################


from HINGE.gene_agreement import compute_gene_agreement_population_2

def run_hinge_2(
    xy_panel1,
    xy_panel2_reg,
    morph_panel1,           # NOT PCA compressed
    morph_panel2,           # NOT PCA compressed
    expr_panel1,            # NOT PCA compressed
    expr_panel2,            # NOT PCA compressed
    MAX_DIST              = None,
    k_candidates          = None,
    threshold             = None,
    N_PCA_MORPH           = 256,
    n_gene_comps          = 128,
    # ── cluster structure parameters (new vs run_hinge) ──
    leiden_resolution     = 0.5,
    n_pca_comps_cluster   = 30,
    n_neighbors_transfer  = 15,
    min_cells_per_cluster = 20,
    verbose = False
):
    """
    HINGE variant that uses cluster-based pseudobulk structure scores
    for gene selection and weighting instead of population-level cv_score.

    Differences from run_hinge
    --------------------------
    compute_gene_agreement_population is called with:
        use_cluster_structure = True   (new)
        min_detection_rate    = 0.02   (relaxed from 0.05)
    This means:
      • Rare cell type markers with low detection are preserved if their
        cluster pattern is reproduced in Panel 2.
      • Genes with technical dropout in one panel receive low structure_score
        and are naturally down-weighted.
      • w_expr is derived from structure-aware agreement scores, so it
        decreases more aggressively when expression is unreliable.
    Everything else (PCA, feature prep, match_combined) is identical.
    """
    if MAX_DIST is None:

        nn_dist = calculate_nn_dist(xy_panel1, xy_panel2_reg)
        p90 = float(np.percentile(nn_dist, 90))
        MAX_DIST = round(p90)
        if verbose:
            print("MAX_DIST is calculated automatically")
            print(f"Using MAX DIST: {MAX_DIST}")

    overlapping_genes = get_overlapping_genes([expr_panel1], [expr_panel2])

    if verbose:
        print(f"  Overlapping genes across all samples: {len(overlapping_genes)}")

    # ── Key difference: cluster-based gene agreement ──────────────────────────
    gene_stats = compute_gene_agreement_population_2(
        expr_panel1,
        expr_panel2,
        overlapping_genes,
        min_detection_rate    = 0.02,       # relaxed: rare markers preserved
        max_log2fc            = 1.0,
        use_cluster_structure = True,        # NEW vs run_hinge
        leiden_resolution     = leiden_resolution,
        n_pca_comps           = n_pca_comps_cluster,
        min_cells_per_cluster = min_cells_per_cluster,
        verbose               = verbose,
    )

    weighted_expr_p1, weighted_expr_p2, selected_genes, w_expr = \
        build_weighted_expression_features(
            expr_panel1,
            expr_panel2,
            gene_stats,
            min_r          = 0.0,
            top_n_genes    = None,
            normalize_expr = False,
            verbose        = verbose,
        )

    # ── Feature preparation (identical to run_hinge) ──────────────────────────
    morph_p1_extracted_list, morph_p2_extracted_list, morph_pca = \
        prepare_morphology_features(
            [morph_panel1], [morph_panel2], N_PCA_MORPH=N_PCA_MORPH
        )

    weighted_expr_p1_adata = ad.AnnData(X=weighted_expr_p1)
    weighted_expr_p1_adata.var_names = selected_genes

    weighted_expr_p2_adata = ad.AnnData(X=weighted_expr_p2)
    weighted_expr_p2_adata.var_names = selected_genes

    w_expr_extracted_p1_list, w_expr_extracted_p2_list, expr_pca, _ = \
        prepare_expression_features(
            [weighted_expr_p1_adata],
            [weighted_expr_p2_adata],
            n_comps=n_gene_comps,
        )

    return match_combined(
        morph_p1_extracted_list[0],
        morph_p2_extracted_list[0],
        w_expr_extracted_p1_list[0],
        w_expr_extracted_p2_list[0],
        xy_panel1,
        xy_panel2_reg,
        max_dist     = MAX_DIST,
        w1           = 1 - w_expr,
        w2           = w_expr,
        k_candidates = k_candidates,
        threshold    = threshold,
    )

##################################################################################
from HINGE.morphology_graph import prepare_morphology_features_graph


# ─────────────────────────────────────────────────────────────────────────────
#  run_hinge_3
# ─────────────────────────────────────────────────────────────────────────────

def run_hinge_3(
    xy_panel1,
    xy_panel2_reg,
    morph_panel1,           # NOT PCA compressed
    morph_panel2,           # NOT PCA compressed
    expr_panel1,            # NOT PCA compressed
    expr_panel2,            # NOT PCA compressed
    MAX_DIST       = None,
    k_candidates   = None,
    threshold      = None,
    N_PCA_MORPH    = 512,
    n_gene_comps   = 512,
    # ── graph smoothing parameters ───────────────────────────────────────────
    k_spatial      = 15,    # spatial neighbours per cell
    n_smooth_iters = 1,     # convolution iterations
    self_weight    = 0.5,   # own feature weight (1-self_weight = neighbour weight)
    max_radius     = None,  # µm; None = use all k_spatial regardless of dist
    verbose        = False,
):
    """
    HINGE with spatial graph-smoothed morphology features.

    Only difference from E_HINGE
    ------------------------------
    prepare_morphology_features  →  prepare_morphology_features_graph

    The graph is built independently on each panel's own spatial coordinates
    (no spatial correspondence assumed).  Each cell's feature becomes a
    weighted mixture of its own PCA feature and the mean of its k_spatial
    nearest spatial neighbours, repeated n_smooth_iters times.

    Recommended parameter starting points
    ---------------------------------------
    k_spatial=15, n_smooth_iters=1, self_weight=0.5
        Conservative: single-hop neighbourhood, equal weighting.
        Best for heterogeneous tissue (sharp boundaries between regions).

    k_spatial=20, n_smooth_iters=2, self_weight=0.5
        Aggressive: two-hop neighbourhood, wider context.
        Best for spatially structured tissue (tumour nests, follicles).

    k_spatial=15, n_smooth_iters=1, self_weight=0.7
        Individual-dominant: less context, closer to E_HINGE.
        Good baseline to isolate neighbourhood contribution.
    """
    if MAX_DIST is None:
        nn_dist  = calculate_nn_dist(xy_panel1, xy_panel2_reg)
        MAX_DIST = round(float(np.percentile(nn_dist, 90)))
        if verbose:
            print(f"  MAX_DIST auto: {MAX_DIST} µm")

    overlapping_genes = get_overlapping_genes([expr_panel1], [expr_panel2])
    if verbose:
        print(f"  Overlapping genes: {len(overlapping_genes)}")

    # ── Expression features (identical to E_HINGE) ────────────────────────────
    gene_stats = compute_gene_agreement_population(
        expr_panel1, expr_panel2, overlapping_genes,
        min_detection_rate = 0.05,
        max_log2fc         = 1.0,
        verbose            = True,
    )

    weighted_expr_p1, weighted_expr_p2, selected_genes, w_expr = \
        build_weighted_expression_features(
            expr_panel1, expr_panel2, gene_stats,
            verbose=True,
        )

    # ── Graph-smoothed morphology features (NEW) ──────────────────────────────
    morph_p1_list, morph_p2_list, _ = prepare_morphology_features_graph(
        [morph_panel1], [morph_panel2],
        [xy_panel1],    [xy_panel2_reg],
        N_PCA_MORPH    = N_PCA_MORPH,
        k_neighbors    = k_spatial,
        n_iterations   = n_smooth_iters,
        self_weight    = self_weight,
        max_radius     = max_radius,
        verbose        = verbose,
    )

    # ── Expression PCA (identical to E_HINGE) ─────────────────────────────────
    weighted_expr_p1_adata = ad.AnnData(X=weighted_expr_p1)
    weighted_expr_p1_adata.var_names = selected_genes

    weighted_expr_p2_adata = ad.AnnData(X=weighted_expr_p2)
    weighted_expr_p2_adata.var_names = selected_genes

    w_expr_p1_list, w_expr_p2_list, _, _ = prepare_expression_features(
        [weighted_expr_p1_adata],
        [weighted_expr_p2_adata],
        n_comps=n_gene_comps,
    )

    return match_combined(
        morph_p1_list[0],
        morph_p2_list[0],
        w_expr_p1_list[0],
        w_expr_p2_list[0],
        xy_panel1,
        xy_panel2_reg,
        max_dist     = MAX_DIST,
        w1           = 1 - w_expr,
        w2           = w_expr,
        k_candidates = k_candidates,
        threshold    = threshold,
    )




####################################################################################################
# hinge 4

# ─────────────────────────────────────────────────────────────────────────────
#  Modified _global_assignment with cluster-aware soft penalty
# ─────────────────────────────────────────────────────────────────────────────

def _global_assignment_cluster_aware(
    xy_panel1,
    xy_panel2_reg,
    score_fn,
    max_dist          = 50.0,
    threshold         = None,
    k_candidates      = None,
    # ── new parameters ────────────────────────────────────────────────────────
    cluster_labels_p1 = None,   # (n1,) int — cluster per Panel 1 cell
    cluster_labels_p2 = None,   # (n2,) int — cluster per Panel 2 cell
    cross_cluster_penalty = 0.3, # subtract this from score for cross-cluster pairs
                                 # 0.0 = no penalty (identical to original)
                                 # 0.3 = moderate discouragement
                                 # 1.0 = near-hard constraint
):
    """
    Maximum-weight one-to-one matching with optional cluster-aware soft penalty.

    For cross-cluster pairs:
        score = score_fn(i, j) - cross_cluster_penalty

    Because scores are cosine similarities in [−1, 1], a penalty of 0.3
    means a cross-cluster pair needs to be 0.3 more similar than a
    same-cluster pair to be selected. This is a strong but not absolute
    discouragement.

    Setting cross_cluster_penalty=0.0 makes this identical to
    _global_assignment.
    """
    n1 = len(xy_panel1)
    n2 = len(xy_panel2_reg)

    if n1 == 0 or n2 == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=float),
        )

    tree = cKDTree(xy_panel2_reg)

    if k_candidates is None:
        candidate_lists = tree.query_ball_point(xy_panel1, r=max_dist)
    else:
        k = min(k_candidates, n2)
        dist, idx = tree.query(xy_panel1, k=k,
                               distance_upper_bound=max_dist)
        if k == 1:
            dist = dist[:, None]
            idx  = idx[:, None]
        candidate_lists = [
            idx[i][idx[i] < n2].tolist() for i in range(n1)
        ]

    scores = np.full((n1, n2 + n1), -1e12, dtype=float)
    scores[:, n2:] = 0.0   # dummy unmatched columns

    use_cluster = (
        cluster_labels_p1 is not None and
        cluster_labels_p2 is not None and
        cross_cluster_penalty > 0.0
    )

    for i, candidates in enumerate(candidate_lists):
        for j in candidates:
            score = float(score_fn(i, j))

            if threshold is not None and score < threshold:
                continue

            # ── Soft cluster penalty ──────────────────────────────────────────
            if use_cluster and cluster_labels_p1[i] != cluster_labels_p2[j]:
                score -= cross_cluster_penalty

            scores[i, j] = score

    row_ind, col_ind = linear_sum_assignment(scores, maximize=True)

    real           = col_ind < n2
    matched_panel1 = row_ind[real]
    matched_panel2 = col_ind[real]
    matched_scores = scores[matched_panel1, matched_panel2]
    valid          = matched_scores > -1e11

    return matched_panel1[valid], matched_panel2[valid], matched_scores[valid]

# ─────────────────────────────────────────────────────────────────────────────
#  run_hinge_4
# ─────────────────────────────────────────────────────────────────────────────


from HINGE.morphology_cluster import compute_morph_cluster_labels
def run_hinge_4(
    xy_panel1,
    xy_panel2_reg,
    morph_panel1,               # NOT PCA compressed (raw UNI AnnData)
    morph_panel2,               # NOT PCA compressed (raw UNI AnnData)
    expr_panel1,
    expr_panel2,
    MAX_DIST              = None,
    k_candidates          = None,
    threshold             = None,
    N_PCA_MORPH           = 512,
    n_gene_comps          = 512,
    # ── cluster parameters ──────────────────────────────────────────────────
    leiden_resolution_morph   = 1.0,   # higher = more clusters
    n_pca_comps_morph_cluster = 50,
    n_neighbors_transfer      = 15,
    min_cluster_size          = 10,
    cross_cluster_penalty     = 0.15,   # tune: 0.1=mild, 0.3=moderate, 0.5=strong
    verbose = True,
):
    """
    HINGE with cluster-aware matching.

    Key addition vs E_HINGE
    -----------------------
    1. UNI features are clustered at high resolution (leiden_resolution=1.0)
       giving ~20-30 clusters for ~8k cells.
    2. Labels are transferred to Panel 2 via cosine KNN on raw UNI features.
    3. During linear_sum_assignment, cross-cluster pairs receive a score
       penalty of `cross_cluster_penalty`, discouraging rare cells from
       being matched to cells of different morphological type.

    This does NOT use spatial coordinates for clustering → no circularity.
    This does NOT require pre-labeled cell types → works on any dataset.
    """
    if MAX_DIST is None:
        nn_dist  = calculate_nn_dist(xy_panel1, xy_panel2_reg)
        MAX_DIST = round(float(np.percentile(nn_dist, 90)))
        if verbose:
            print(f"  MAX_DIST auto: {MAX_DIST} µm")

    # ── Step 1: Compute morphological cluster labels ──────────────────────────
    if verbose:
        print("\n  Computing morphological cluster labels...")

    labels_p1, labels_p2, cluster_info = compute_morph_cluster_labels(
        morph_panel1,
        morph_panel2,
        n_pca_comps           = n_pca_comps_morph_cluster,
        leiden_resolution     = leiden_resolution_morph,
        n_neighbors_transfer  = n_neighbors_transfer,
        min_cluster_size      = min_cluster_size,
        verbose               = verbose,
    )

    if verbose:
        print(f"\n  Morphological clusters: {cluster_info['K']}  "
              f"(cross-cluster penalty = {cross_cluster_penalty})")

    # ── Step 2: Expression features (identical to E_HINGE) ───────────────────
    overlapping_genes = get_overlapping_genes([expr_panel1], [expr_panel2])

    gene_stats = compute_gene_agreement_population(
        expr_panel1, expr_panel2, overlapping_genes,
        min_detection_rate = 0.05,
        max_log2fc         = 1.0,
        verbose            = verbose,
    )

    weighted_expr_p1, weighted_expr_p2, selected_genes, w_expr = \
        build_weighted_expression_features(
            expr_panel1, expr_panel2, gene_stats,
            verbose=verbose,
        )

    # ── Step 3: Morphology PCA (identical to E_HINGE) ────────────────────────
    morph_p1_list, morph_p2_list, _ = prepare_morphology_features(
        [morph_panel1], [morph_panel2], N_PCA_MORPH=N_PCA_MORPH
    )

    # ── Step 4: Expression PCA ────────────────────────────────────────────────
    weighted_expr_p1_adata = ad.AnnData(X=weighted_expr_p1)
    weighted_expr_p1_adata.var_names = selected_genes
    weighted_expr_p2_adata = ad.AnnData(X=weighted_expr_p2)
    weighted_expr_p2_adata.var_names = selected_genes

    w_expr_p1_list, w_expr_p2_list, _, _ = prepare_expression_features(
        [weighted_expr_p1_adata], [weighted_expr_p2_adata],
        n_comps=n_gene_comps,
    )

    morph_p1 = morph_p1_list[0]
    morph_p2 = morph_p2_list[0]
    expr_p1  = w_expr_p1_list[0]
    expr_p2  = w_expr_p2_list[0]
    w_morph  = 1.0 - w_expr

    # ── Step 5: Cluster-aware combined matching ───────────────────────────────
    def combined_score(i, j):
        s_morph = float(morph_p1[i] @ morph_p2[j])
        s_expr  = float(expr_p1[i]  @ expr_p2[j])
        return w_morph * s_morph + w_expr * s_expr

    matched_p1, matched_p2, matched_sc = _global_assignment_cluster_aware(
        xy_panel1,
        xy_panel2_reg,
        combined_score,
        max_dist              = MAX_DIST,
        threshold             = threshold,
        k_candidates          = k_candidates,
        cluster_labels_p1     = labels_p1,
        cluster_labels_p2     = labels_p2,
        cross_cluster_penalty = cross_cluster_penalty,
    )

    matched_dist = np.linalg.norm(
        xy_panel1[matched_p1] - xy_panel2_reg[matched_p2], axis=1
    )
    if verbose:
        print(
            f"  [H HINGE4     ] {len(matched_p1):>6} pairs "
            f"({len(matched_p1)/len(xy_panel1):.1%}), "
            f"median dist = {np.median(matched_dist):.2f}, "
            f"median sim = {np.median(matched_sc):.3f}"
        )

    return matched_p1, matched_p2

#########################################################################################################################

from HINGE.morphology_cluster import compute_morph_cluster_labels_joint

def run_hinge_5(
    xy_panel1,
    xy_panel2_reg,
    morph_panel1,               # raw UNI (NOT PCA-compressed)
    morph_panel2,               # raw UNI (NOT PCA-compressed)
    expr_panel1,
    expr_panel2,
    MAX_DIST                  = None,
    k_candidates              = None,
    threshold                 = None,
    N_PCA_MORPH               = 512,
    n_gene_comps              = 512,
    # ── cluster parameters ──────────────────────────────────────────────────
    leiden_resolution_morph   = 1.0,
    n_pca_comps_morph_cluster = 50,
    min_cluster_size          = 10,
    cross_cluster_penalty     = 0.15,
    verbose                   = False,
):
    """
    HINGE5 — joint morphology clustering + cluster-aware matching.

    Difference from HINGE4
    ----------------------
    HINGE4: cluster P1 → KNN-transfer labels to P2
    HINGE5: concatenate P1+P2 → joint Leiden → split labels back

    Joint clustering is more principled because:
    - P2 cells near cluster boundaries are assigned directly, not estimated
    - No KNN hyperparameter (k_transfer) to tune
    - Panel-mixing diagnostic reveals if batch effects exist in UNI space

    Everything else (expression features, combined scoring,
    _global_assignment_cluster_aware) is identical to HINGE4.
    """
    if MAX_DIST is None:
        nn_dist  = calculate_nn_dist(xy_panel1, xy_panel2_reg)
        MAX_DIST = round(float(np.percentile(nn_dist, 95)))
        if verbose:
            print(f"  MAX_DIST auto: {MAX_DIST} µm")

    # ── Step 1: Joint morphological clustering ────────────────────────────────
    if verbose:
        print("\n  [I_HINGE5] Computing JOINT morphological cluster labels...")

    labels_p1, labels_p2, cluster_info = compute_morph_cluster_labels_joint(
        morph_panel1,
        morph_panel2,
        n_pca_comps       = n_pca_comps_morph_cluster,
        leiden_resolution = leiden_resolution_morph,
        min_cluster_size  = min_cluster_size,
        verbose           = verbose,
    )

    if verbose:
        print(f"\n  Morphological clusters (joint): {cluster_info['K']}  "
              f"(cross-cluster penalty = {cross_cluster_penalty})")

    # ── Step 2: Expression features (identical to E_HINGE / HINGE4) ──────────
    overlapping_genes = get_overlapping_genes([expr_panel1], [expr_panel2])

    gene_stats = compute_gene_agreement_population(
        expr_panel1, expr_panel2, overlapping_genes,
        min_detection_rate = 0.05,
        max_log2fc         = 1.0,
        verbose            = verbose,
    )

    weighted_expr_p1, weighted_expr_p2, selected_genes, w_expr = \
        build_weighted_expression_features(
            expr_panel1, expr_panel2, gene_stats,
            verbose=verbose,
        )

    # ── Step 3: Morphology PCA ────────────────────────────────────────────────
    morph_p1_list, morph_p2_list, _ = prepare_morphology_features(
        [morph_panel1], [morph_panel2], N_PCA_MORPH=N_PCA_MORPH
    )

    # ── Step 4: Expression PCA ────────────────────────────────────────────────
    weighted_expr_p1_adata = ad.AnnData(X=weighted_expr_p1)
    weighted_expr_p1_adata.var_names = selected_genes
    weighted_expr_p2_adata = ad.AnnData(X=weighted_expr_p2)
    weighted_expr_p2_adata.var_names = selected_genes

    w_expr_p1_list, w_expr_p2_list, _, _ = prepare_expression_features(
        [weighted_expr_p1_adata], [weighted_expr_p2_adata],
        n_comps=n_gene_comps,
    )

    morph_p1 = morph_p1_list[0]
    morph_p2 = morph_p2_list[0]
    expr_p1  = w_expr_p1_list[0]
    expr_p2  = w_expr_p2_list[0]
    w_morph  = 1.0 - w_expr

    # ── Step 5: Cluster-aware combined matching ───────────────────────────────
    def combined_score(i, j):
        s_morph = float(morph_p1[i] @ morph_p2[j])
        s_expr  = float(expr_p1[i]  @ expr_p2[j])
        return w_morph * s_morph + w_expr * s_expr

    matched_p1, matched_p2, matched_sc = _global_assignment_cluster_aware(
        xy_panel1,
        xy_panel2_reg,
        combined_score,
        max_dist              = MAX_DIST,
        threshold             = threshold,
        k_candidates          = k_candidates,
        cluster_labels_p1     = labels_p1,
        cluster_labels_p2     = labels_p2,
        cross_cluster_penalty = cross_cluster_penalty,
    )

    matched_dist = np.linalg.norm(
        xy_panel1[matched_p1] - xy_panel2_reg[matched_p2], axis=1
    )
    if verbose:
        print(
            f"  [I HINGE5     ] {len(matched_p1):>6} pairs "
            f"({len(matched_p1)/len(xy_panel1):.1%}), "
            f"median dist = {np.median(matched_dist):.2f}, "
            f"median sim = {np.median(matched_sc):.3f}"
        )

    return matched_p1, matched_p2


from HINGE.morphology_cluster import compute_morph_cluster_labels_joint_harmonized

def run_hinge_6(
    xy_panel1,
    xy_panel2_reg,
    morph_panel1,
    morph_panel2,
    expr_panel1,
    expr_panel2,
    MAX_DIST                  = None,
    k_candidates              = None,
    threshold                 = None,
    N_PCA_MORPH               = 512,
    n_gene_comps              = 512,
    # ── cluster parameters ──────────────────────────────────────────────────
    leiden_resolution_morph   = 1.0,
    n_pca_comps_morph_cluster = 50,
    min_cluster_size          = 10,
    cross_cluster_penalty     = 0.15,
    # ── harmonization parameters ─────────────────────────────────────────────
    harmony_method            = "harmony",  # "harmony" | "mean_center" | "none"
    n_harmony_iters           = 10,
    harmony_theta             = 2.0,
    verbose                   = True,
):
    """
    HINGE6 — harmonized joint morphology clustering + cluster-aware matching.

    Difference from HINGE5
    ----------------------
    HINGE5: joint Leiden on raw joint PCA
    HINGE6: joint Leiden on HARMONY-corrected joint PCA

    This directly addresses the panel separation failure mode seen in HINGE5
    at alpha >= 0.75 (mixing std 0.21–0.40), where synthetic noise caused
    P1 and P2 to cluster separately. Harmony removes the inter-panel centroid
    shift while preserving within-panel biological structure.

    In real data, the correction is minimal (|delta_mean| = 0.00264) but
    harmless — HINGE6 degrades gracefully to HINGE5 when panels are already
    well-mixed.
    """
    if MAX_DIST is None:
        nn_dist  = calculate_nn_dist(xy_panel1, xy_panel2_reg)
        MAX_DIST = round(float(np.percentile(nn_dist, 95)))
        if verbose:
            print(f"  MAX_DIST auto: {MAX_DIST} µm")

    # ── Step 1: Harmonized joint morphological clustering ─────────────────────
    if verbose:
        print(f"\n  [J_HINGE6] Computing HARMONIZED joint morphological clusters "
              f"(method={harmony_method})...")

    labels_p1, labels_p2, cluster_info = compute_morph_cluster_labels_joint_harmonized(
        morph_panel1,
        morph_panel2,
        n_pca_comps       = n_pca_comps_morph_cluster,
        leiden_resolution = leiden_resolution_morph,
        min_cluster_size  = min_cluster_size,
        harmony_method    = harmony_method,
        n_harmony_iters   = n_harmony_iters,
        harmony_theta     = harmony_theta,
        verbose           = verbose,
    )

    if verbose:
        print(f"\n  Morphological clusters (harmonized joint): {cluster_info['K']}  "
              f"(cross-cluster penalty = {cross_cluster_penalty})")

    # ── Steps 2–4: Expression + Morphology features (identical to HINGE5) ────
    overlapping_genes = get_overlapping_genes([expr_panel1], [expr_panel2])

    gene_stats = compute_gene_agreement_population(
        expr_panel1, expr_panel2, overlapping_genes,
        min_detection_rate = 0.05,
        max_log2fc         = 1.0,
        verbose            = verbose,
    )

    weighted_expr_p1, weighted_expr_p2, selected_genes, w_expr = \
        build_weighted_expression_features(
            expr_panel1, expr_panel2, gene_stats,
            verbose=verbose,
        )

    morph_p1_list, morph_p2_list, _ = prepare_morphology_features(
        [morph_panel1], [morph_panel2], N_PCA_MORPH=N_PCA_MORPH
    )

    weighted_expr_p1_adata = ad.AnnData(X=weighted_expr_p1)
    weighted_expr_p1_adata.var_names = selected_genes
    weighted_expr_p2_adata = ad.AnnData(X=weighted_expr_p2)
    weighted_expr_p2_adata.var_names = selected_genes

    w_expr_p1_list, w_expr_p2_list, _, _ = prepare_expression_features(
        [weighted_expr_p1_adata], [weighted_expr_p2_adata],
        n_comps=n_gene_comps,
    )

    morph_p1 = morph_p1_list[0]
    morph_p2 = morph_p2_list[0]
    expr_p1  = w_expr_p1_list[0]
    expr_p2  = w_expr_p2_list[0]
    w_morph  = 1.0 - w_expr

    # ── Step 5: Cluster-aware combined matching (identical to HINGE5) ─────────
    def combined_score(i, j):
        s_morph = float(morph_p1[i] @ morph_p2[j])
        s_expr  = float(expr_p1[i]  @ expr_p2[j])
        return w_morph * s_morph + w_expr * s_expr

    matched_p1, matched_p2, matched_sc = _global_assignment_cluster_aware(
        xy_panel1,
        xy_panel2_reg,
        combined_score,
        max_dist              = MAX_DIST,
        threshold             = threshold,
        k_candidates          = k_candidates,
        cluster_labels_p1     = labels_p1,
        cluster_labels_p2     = labels_p2,
        cross_cluster_penalty = cross_cluster_penalty,
    )

    matched_dist = np.linalg.norm(
        xy_panel1[matched_p1] - xy_panel2_reg[matched_p2], axis=1
    )
    if verbose:
        print(
            f"  [J HINGE6     ] {len(matched_p1):>6} pairs "
            f"({len(matched_p1)/len(xy_panel1):.1%}), "
            f"median dist = {np.median(matched_dist):.2f}, "
            f"median sim = {np.median(matched_sc):.3f}"
        )

    return matched_p1, matched_p2

'''
Run all matching strategies (distance, histology, expression, combined) and print results.

'''
def run_all_strategies(
    xy_panel1,
    xy_panel2_reg,
    morph_panel1,
    morph_panel2,
    expr_panel1,
    expr_panel2,
    MAX_DIST=None,
    k_candidates=None,
    pair_label="",
    threshold=None,
    extracted = True,
    N_PCA_MORPH = 256,
    n_gene_comps = 128,
    
    verbose = False,
    
    #for hinge_1
    run_hinge1 = True,
    
    #parameters for hinge_2──
    run_hinge2 = True,
    leiden_resolution     = 0.5,
    n_pca_comps_cluster   = 30,
    n_neighbors_transfer  = 15,
    min_cells_per_cluster = 20,

    
    
    # ── graph smoothing for G_HINGE3 ─────
    run_hinge3 = False,
    k_spatial      = 6,
    n_smooth_iters = 1,
    self_weight    = 0.8,
    #max_radius     = None,
    
    
    # Hinge4
    run_hinge4   = True,
    leiden_resolution_morph   = 1.0,
    cross_cluster_penalty = 0.15,
    
    
    # ── Hinge5 ──────────────────────────────────────────────────────────
    run_hinge5              = False,
    leiden_resolution_morph5 = 1.0,   # separate tunable if needed
    cross_cluster_penalty5   = 0.15,
    
    # ── HINGE6 ──────────────────────────────────────────────────────────
    run_hinge6                = False,
    leiden_resolution_morph6  = 1.0,
    cross_cluster_penalty6    = 0.15,
    harmony_method            = "mean_center",   # "harmony" | "mean_center" | "none"
    n_harmony_iters           = 10,
    harmony_theta             = 2.0,
):
    print(f"\n  --- {pair_label} ---")

    if MAX_DIST is None:
        
        
        nn_dist = calculate_nn_dist(xy_panel1, xy_panel2_reg)

        p50  = float(np.percentile(nn_dist, 50))
        p75  = float(np.percentile(nn_dist, 75))
        p90  = float(np.percentile(nn_dist, 90))
        p95  = float(np.percentile(nn_dist, 95))
        p99  = float(np.percentile(nn_dist, 99))

        if verbose:
            print("MAX_DIST is calculated automatically")
            print(f"  Core NN dist p50     : {p50:.2f} um")
            print(f"  Core NN dist p75     : {p75:.2f} um")
            print(f"  Core NN dist p90     : {p90:.2f} um")
            print(f"  Core NN dist p95     : {p95:.2f} um")
            print(f"  Core NN dist p99     : {p99:.2f} um")
            print(f"Using MAX DIST: {round(p99)}")

    
        MAX_DIST = round(p99)

        
        
        
    if extracted:
        result_dict= {
            "A_distance": match_distance_only(
                xy_panel1,
                xy_panel2_reg,
                MAX_DIST,
                k_candidates,
            ),

            "B_histology": match_feature_only(
                morph_panel1,
                morph_panel2,
                xy_panel1,
                xy_panel2_reg,
                MAX_DIST,
                k_candidates,
                label="B histology",
                threshold=threshold,
            ),

            "C_expression": match_feature_only(
                expr_panel1,
                expr_panel2,
                xy_panel1,
                xy_panel2_reg,
                MAX_DIST,
                k_candidates,
                label="C expression",
                threshold=threshold,
            ),

            "D_combined": match_combined(
                morph_panel1,
                morph_panel2,
                expr_panel1,
                expr_panel2,
                xy_panel1,
                xy_panel2_reg,
                max_dist=MAX_DIST,
                w1=0.5,
                w2=0.5,
                k_candidates=k_candidates,
                threshold=threshold,
            ),
        }
        
    else:
        # ── Prepare features ──────────────────────────────────────────────────────
        morph_p1_extracted_list, morph_pl2_extracted_list, morph_pca = prepare_morphology_features(
            [morph_panel1], [morph_panel2], N_PCA_MORPH=N_PCA_MORPH
        )
    
        expr_extracted_p1_list, expr_extracted_p2_list, expr_pca, overlap_genes = prepare_expression_features(
        [expr_panel1],[expr_panel2], n_comps=n_gene_comps)
        
        morph_extracted_p1 = morph_p1_extracted_list[0]
        morph_extracted_p2 = morph_pl2_extracted_list[0]        
        expr_extracted_p1 = expr_extracted_p1_list[0]
        expr_extracted_p2 = expr_extracted_p2_list[0]
    
        result_dict= {
            "A_distance": match_distance_only(
                xy_panel1,
                xy_panel2_reg,
                MAX_DIST,
                k_candidates,
            ),

            "B_histology": match_feature_only(
                morph_extracted_p1,
                morph_extracted_p2,
                xy_panel1,
                xy_panel2_reg,
                MAX_DIST,
                k_candidates,
                label="B histology",
                threshold=threshold,
            ),

            "C_expression": match_feature_only(
                expr_extracted_p1,
                expr_extracted_p2,
                xy_panel1,
                xy_panel2_reg,
                MAX_DIST,
                k_candidates,
                label="C expression",
                threshold=threshold,
            ),

            "D_combined": match_combined(
                morph_extracted_p1,
                morph_extracted_p2,
                expr_extracted_p1,
                expr_extracted_p2,
                xy_panel1,
                xy_panel2_reg,
                max_dist=MAX_DIST,
                w1=0.5,
                w2=0.5,
                k_candidates=k_candidates,
                threshold=threshold,
            )

        }
        
        
        if run_hinge1:
            result_dict["E_HINGE"] = run_hinge(
                xy_panel1,
                xy_panel2_reg,
                morph_panel1,#NOT PCA COMPRESSED
                morph_panel2,#NOT PCA COMPRESSED
                expr_panel1,#NOT PCA COMPRESSED
                expr_panel2,#NOT PCA COMPRESSED
                MAX_DIST=MAX_DIST,
                k_candidates=None,
                threshold=None,
                N_PCA_MORPH = N_PCA_MORPH,
                n_gene_comps = n_gene_comps
            )
            
        if run_hinge2:
            
                        # ── F_HINGE2: cluster structure-based gene agreement (new) ────────
            result_dict["F_HINGE2"]= run_hinge_2(
                xy_panel1, xy_panel2_reg,
                morph_panel1, morph_panel2,
                expr_panel1, expr_panel2,
                MAX_DIST              = MAX_DIST,
                k_candidates          = None,
                threshold             = None,
                N_PCA_MORPH           = N_PCA_MORPH,
                n_gene_comps          = n_gene_comps,
                leiden_resolution     = leiden_resolution,
                n_pca_comps_cluster   = n_pca_comps_cluster,
                n_neighbors_transfer  = n_neighbors_transfer,
                min_cells_per_cluster = min_cells_per_cluster,
            )

        if run_hinge3:
             result_dict["G_HINGE3"]= run_hinge_3(
            xy_panel1, xy_panel2_reg,
            morph_panel1, morph_panel2,
            expr_panel1, expr_panel2,
            MAX_DIST       = MAX_DIST,
            k_candidates   = None,
            threshold      = None,
            N_PCA_MORPH    = N_PCA_MORPH,
            n_gene_comps   = n_gene_comps,
            k_spatial      = k_spatial,
            n_smooth_iters = n_smooth_iters,
            self_weight    = self_weight,
            max_radius     = float(np.percentile(nn_dist, 25)),
        )
             
        if run_hinge4:
            result_dict["H_HINGE4"]=run_hinge_4(
                xy_panel1, xy_panel2_reg,
                morph_panel1, morph_panel2,
                expr_panel1, expr_panel2,
                MAX_DIST                  = MAX_DIST,
                k_candidates              = None,
                threshold                 = None,
                N_PCA_MORPH               = N_PCA_MORPH,
                n_gene_comps              = n_gene_comps,
                leiden_resolution_morph   = leiden_resolution_morph,
                cross_cluster_penalty     = cross_cluster_penalty,
            )
            
        if run_hinge5:
            result_dict["I_HINGE5"] = run_hinge_5(
                xy_panel1, xy_panel2_reg,
                morph_panel1, morph_panel2,
                expr_panel1, expr_panel2,
                MAX_DIST                  = MAX_DIST,
                k_candidates              = None,
                threshold                 = None,
                N_PCA_MORPH               = N_PCA_MORPH,
                n_gene_comps              = n_gene_comps,
                leiden_resolution_morph   = leiden_resolution_morph5,
                cross_cluster_penalty     = cross_cluster_penalty5,
            )    

        if run_hinge6:
            result_dict["J_HINGE6"] = run_hinge_6(
                xy_panel1, xy_panel2_reg,
                morph_panel1, morph_panel2,
                expr_panel1, expr_panel2,
                MAX_DIST                  = MAX_DIST,
                k_candidates              = None,
                threshold                 = None,
                N_PCA_MORPH               = N_PCA_MORPH,
                n_gene_comps              = n_gene_comps,
                leiden_resolution_morph   = leiden_resolution_morph6,
                cross_cluster_penalty     = cross_cluster_penalty6,
                harmony_method            = harmony_method,
                n_harmony_iters           = n_harmony_iters,
                harmony_theta             = harmony_theta,
            )
        
    return result_dict
    
    
    
    
    