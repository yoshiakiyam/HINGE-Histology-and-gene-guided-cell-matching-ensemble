
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

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
    n_gene_comps = 128
):
    if MAX_DIST is None:
        #automatically set distance
        print("MAX_DIST is calculated automatically")
        
        nn_dist = calculate_nn_dist(xy_panel1, xy_panel2_reg)

        p50  = float(np.percentile(nn_dist, 50))
        p75  = float(np.percentile(nn_dist, 75))
        p90  = float(np.percentile(nn_dist, 90))
        p95  = float(np.percentile(nn_dist, 95))
        p99  = float(np.percentile(nn_dist, 99))
    
        print(f"  Core NN dist p50     : {p50:.2f} um")
        print(f"  Core NN dist p75     : {p75:.2f} um")
        print(f"  Core NN dist p90     : {p90:.2f} um")
        print(f"  Core NN dist p95     : {p95:.2f} um")
        print(f"  Core NN dist p99     : {p99:.2f} um")

    
        MAX_DIST = round(p90)
        print(f"Using MAX DIST: {round(p90)}")
        
        
    
    overlapping_genes = get_overlapping_genes([expr_panel1], [expr_panel2])
    print(f"  Overlapping genes across all samples: {len(overlapping_genes)}")
    
    
    gene_stats = compute_gene_agreement_population(expr_panel1, expr_panel2, overlapping_genes,
                                                   min_detection_rate=0.05,
                                                   max_log2fc=1.0,
                                                   verbose = True)
    
    weighted_expr_p1, weighted_expr_p2, _, w_expr = build_weighted_expression_features(
        expr_panel1,
        expr_panel2,
        gene_stats,
        min_r          = 0.0,      # min agreement_score to include in the gene to use
        top_n_genes    = None,
        normalize_expr = False,
        verbose        = True,
    )
    
    
    # ── Prepare features ──────────────────────────────────────────────────────
    morph_p1_extracted_list, morph_pl2_extracted_list, morph_pca = prepare_morphology_features(
        [morph_panel1], [morph_panel2], N_PCA_MORPH=N_PCA_MORPH
    )
    
    w_expr_extracted_p1_list, w_expr_extracted_p2_list, expr_pca, overlap_genes = prepare_expression_features(
    [weighted_expr_p1],[weighted_expr_p2], n_comps=n_gene_comps)
    
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
    n_gene_comps = 128
):
    print(f"\n  --- {pair_label} ---")

    if MAX_DIST is None:
        print("MAX_DIST is calculated automatically")
        
        nn_dist = calculate_nn_dist(xy_panel1, xy_panel2_reg)

        p50  = float(np.percentile(nn_dist, 50))
        p75  = float(np.percentile(nn_dist, 75))
        p90  = float(np.percentile(nn_dist, 90))
        p95  = float(np.percentile(nn_dist, 95))
        p99  = float(np.percentile(nn_dist, 99))
    
        print(f"  Core NN dist p50     : {p50:.2f} um")
        print(f"  Core NN dist p75     : {p75:.2f} um")
        print(f"  Core NN dist p90     : {p90:.2f} um")
        print(f"  Core NN dist p95     : {p95:.2f} um")
        print(f"  Core NN dist p99     : {p99:.2f} um")

    
        MAX_DIST = round(p90)
        print(f"Using MAX DIST: {round(p90)}")
        
        
        
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
            ),
            "E_HINGE": run_hinge(
                xy_panel1,
                xy_panel2_reg,
                morph_panel1,#NOT PCA COMPRESSED
                morph_panel2,#NOT PCA COMPRESSED
                expr_panel1,#NOT PCA COMPRESSED
                expr_panel2,#NOT PCA COMPRESSED
                MAX_DIST=None,
                k_candidates=None,
                threshold=None,
                N_PCA_MORPH = N_PCA_MORPH,
                n_gene_comps = n_gene_comps
            )
        }
        
    return result_dict
    
    
    
    
    