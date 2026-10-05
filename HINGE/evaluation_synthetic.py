

import numpy as np

from pathlib import Path
import pandas as pd

from HINGE.synthetic_data_whole import create_synthetic_sample,create_synthetic_replicates
from HINGE.utils import compute_population_diff_stats
from HINGE.matching_with_global_assignment import run_all_strategies
from HINGE.utils import _uni_matrix, _xy, get_expr_pca
from HINGE.feature_mapping import prepare_morphology_features









########################################################################################################
#synthetic evaluation



def evaluate_synthetic(
    s_orig,
    s_t,
    morph_orig,
    morph_t,
    expr_orig,
    expr_t,
    MAX_DIST=None,
    k_candidates=20,
    pair_label="",
    spatial_key="spatial",
    threshold=None,
    extracted = True,
    N_PCA_MORPH = 256,
    n_gene_comps = 128
):
    """
    Match s_orig to fake section s_t. Accuracy is the fraction of matched
    pairs whose panel2 index equals the source index (same cell id / order).
    """
    xy_orig = _xy(s_orig, spatial_key)
    xy_t = s_t["xy"]
    gt = s_t["ground_truth"]
    n = len(xy_orig)



    match_results = run_all_strategies(
        xy_panel1=xy_orig,
        xy_panel2_reg=xy_t,
        morph_panel1=_uni_matrix(morph_orig),
        morph_panel2=morph_t,
        expr_panel1=expr_orig,
        expr_panel2=expr_t,
        MAX_DIST=MAX_DIST,
        k_candidates=k_candidates,
        pair_label=pair_label,
        threshold=threshold,
        extracted=extracted,
        N_PCA_MORPH=N_PCA_MORPH,
        n_gene_comps=n_gene_comps,
    )

    results = {}
    print(f"\n  {'Strategy':<20} {'Matched':>8} {'Correct':>8} "
          f"{'Acc_match':>10} {'Acc_all':>10} {'Coverage':>10}")
    print(f"  {'-' * 70}")

    for strategy, (idx_p1, idx_p2) in match_results.items():
        
        #if no matches were found, set accuracy and coverage to 0
        if len(idx_p1) == 0:
            results[strategy] = {
                "accuracy": 0.0,
                "accuracy_all": 0.0,
                "coverage": 0.0,
                "n_matched": 0,
                "n_correct": 0,
            }
            print(f"  {strategy:<20} {0:>8} {0:>8} "
                  f"{0:>10.3f} {0:>10.3f} {0:>10.3f}")
            continue
        
        #compute the number of correct matches, 
        # accuracy among matched pairs, 
        # accuracy among all pairs, 
        # and coverage
        
        correct = (idx_p2 == gt[idx_p1]).sum()
        accuracy = correct / len(idx_p1)
        accuracy_all = correct / n
        coverage = len(idx_p1) / n
        
        #store the results for this strategy in the results dictionary
        results[strategy] = {
            "accuracy": float(accuracy),
            "accuracy_all": float(accuracy_all),
            "coverage": float(coverage),
            "n_matched": int(len(idx_p1)),
            "n_correct": int(correct),
            "idx_p1": idx_p1,
            "idx_p2": idx_p2,
        }
        print(
            f"  {strategy:<20} {len(idx_p1):>8} {correct:>8} "
            f"{accuracy:>10.3f} {accuracy_all:>10.3f} {coverage:>10.3f}"
        )

    return results


from HINGE.feature_mapping import prepare_expression_features
from sklearn.preprocessing import normalize
def run_synthetic_evaluation(
    adata_panel1,
    uni_panel1_normed,
    adata_panel2,
    uni_panel2_normed,

    expr_panel1,
    overlap_genes,    
    spatial_key_p1    = "spatial",
    spatial_key_p2    = "spatial_registered",
    alpha_list        = [0.25, 0.5, 0.75, 1.0],
    n_seeds_per_alpha = 3,
    noise_scale       = 1.0,
    MAX_DIST          = None,
    k_candidates      = 20,
    SAMPLE_NAME       = "s1", #For labels
    RECENTER          = False, #recentes the fake cells after created
    NORMALIZE_EXPR    = False, #True if the original xenium data is not normalized
    xy_method="displacement_field",
    xy_k=20,
    xy_local_noise_weight=0.25,
    
    threshold=None,
    N_PCA_MORPH = 256,
    n_gene_comps = 128,
    run_hinge= False # if true, activate the HINGE matching strategy
):
    
    """
    Full pipeline:
        1. Compute population diff stats (panel1 vs panel2)
        2. Create synthetic replicates of panel1
        3. Run all matching strategies on each replicate
        4. Return accuracy DataFrame
    """
    # Step 1: stats
    stats = compute_population_diff_stats(
        adata_panel1   = adata_panel1,
        uni_panel1_normed     = uni_panel1_normed,
        adata_panel2   = adata_panel2,
        uni_panel2_normed     = uni_panel2_normed,
        overlap_genes  = overlap_genes,
        spatial_key_p1 = spatial_key_p1,
        spatial_key_p2 = spatial_key_p2,
    )

    # Step 2: replicates
    fakes = create_synthetic_replicates(
        adata_panel1      = adata_panel1,
        uni_panel1_normed = uni_panel1_normed,
        stats             = stats,
        alpha_list        = alpha_list,
        n_seeds_per_alpha = n_seeds_per_alpha,
        noise_scale       = noise_scale,
        spatial_key       = spatial_key_p1,
        RECENTER          = RECENTER,
        xy_method=xy_method,
        xy_k=xy_k,
        xy_local_noise_weight=xy_local_noise_weight,
    )

    # Step 3: evaluate
    strategies = ["A_distance", "B_histology", "C_expression", "D_combined"]
    records    = []

    for fake in fakes:
        rep   = fake['replicate']
        alpha = fake['alpha']
        n1    = len(adata_panel1)

        print(f"\n{'='*58}")
        print(f"  {SAMPLE_NAME} → {SAMPLE_NAME}_t  "
              f"[alpha={alpha:.2f}  rep={rep+1}]")
        print(f"{'='*58}")
        
        
        if run_hinge:
            eval_results = evaluate_synthetic(
                s_orig         = adata_panel1,
                s_t            = fake,
                
                morph_orig     = morph_panel1_list[0],
                morph_t        = morph_fake_list[0],
                
                expr_orig      = expr_panel1,
                expr_t         = fake["adata"],
                
                MAX_DIST       = MAX_DIST,
                k_candidates   = k_candidates,
                pair_label     = f"{SAMPLE_NAME} α={alpha:.2f} rep={rep+1}",
                spatial_key    = spatial_key_p1,
                
                threshold      = threshold,
                extracted      = True,
                N_PCA_MORPH    = N_PCA_MORPH,
                n_gene_comps   = n_gene_comps,
            )
        
        if not run_hinge:
        
            
            # ── Prepare features ──────────────────────────────────────────────────────
            morph_panel1_list, morph_fake_list, morph_pca = prepare_morphology_features(
                [uni_panel1_normed], [fake["uni"]], N_PCA_MORPH=N_PCA_MORPH
            )
        
            expr_extracted_p1_list, expr_extracted_p2_list, expr_pca, overlap_genes = prepare_expression_features(
            [expr_panel1],[fake["adata"]], n_comps=n_gene_comps)
            

            #print(f"expr_pca type: {type(expr_pca)}   expr_t: {type(expr_t)} ")
            
            #add feature extraction step

            eval_results = evaluate_synthetic(
                s_orig         = adata_panel1,
                s_t            = fake,
                
                morph_orig     = morph_panel1_list[0],
                morph_t        = morph_fake_list[0],
                
                expr_orig      = expr_extracted_p1_list[0],
                expr_t         = expr_extracted_p2_list[0],
                
                MAX_DIST       = MAX_DIST,
                k_candidates   = k_candidates,
                pair_label     = f"{SAMPLE_NAME} α={alpha:.2f} rep={rep+1}",
                spatial_key    = spatial_key_p1,
                
                threshold      = threshold,
                extracted      = True,
                N_PCA_MORPH    = N_PCA_MORPH,
                n_gene_comps   = n_gene_comps,
            )
            
            

        for strategy in strategies:
            rec = eval_results[strategy]
            records.append({
                "sample"       : SAMPLE_NAME,
                "alpha"        : alpha,
                "replicate"    : rep + 1,
                "strategy"     : strategy,
                "accuracy"     : rec["accuracy"],
                "accuracy_all" : rec["accuracy_all"],
                "coverage"     : rec["coverage"],
                "n_matched"    : rec["n_matched"],
                "n_correct"    : rec["n_correct"],
                "n_total"      : n1,
            })

    return pd.DataFrame(records), fakes, stats







###################################################################################################################
#synthetic with dropout


def apply_dropout(
    adata_panel1,
    dropout_frac    = 0.10,
    dropout_mode    = "independent",  # "independent" | "symmetric"
    random_seed     = 999,
):
    """
    Apply cell dropout to both s1 (source) and s1_t (synthetic target).

    Parameters
    ----------
    fake         : dict    output of create_synthetic_sample()
    adata_panel1 : AnnData real panel1 (s1)
    dropout_frac : float   fraction of cells to drop from each panel
    dropout_mode :
        "independent" → drop different random cells from s1 and s1_t
                        simulates independent cell loss in each section
                        most realistic for adjacent sections
        "symmetric"   → drop the same cells from both
                        simulates cells absent from the tissue region entirely
                        (e.g. tissue folds, section tears)
    random_seed  : int

    Returns
    -------
    dropout_result : dict with keys:
        's1_keep_idx'   : (n_keep,) indices into adata_panel1 that remain
        's1t_keep_idx'  : (n_keep_t,) indices into fake['xy'] that remain
        'eval_idx'      : (n_eval,) indices into s1_keep_idx that have
                          their true match still present in s1_t
                          ← ONLY these cells are used for accuracy evaluation
        's1_drop_idx'   : dropped from s1
        's1t_drop_idx'  : dropped from s1_t
        'dropout_frac'  : float
        'dropout_mode'  : str
        'n_original'    : int
        'n_s1_keep'     : int
        'n_s1t_keep'    : int
        'n_eval'        : int   cells with valid ground truth available
    """
    rng      = np.random.default_rng(random_seed)
    n        = len(adata_panel1)
    n_drop   = int(np.round(dropout_frac * n))

    if dropout_mode == "independent":
        # Drop different cells from s1 and s1_t
        s1_drop_idx  = rng.choice(n, size=n_drop, replace=False)
        s1t_drop_idx = rng.choice(n, size=n_drop, replace=False)

    elif dropout_mode == "symmetric":
        # Drop the same cells from both
        drop_idx     = rng.choice(n, size=n_drop, replace=False)
        s1_drop_idx  = drop_idx
        s1t_drop_idx = drop_idx

    else:
        raise ValueError(f"Unknown dropout_mode: {dropout_mode}")

    
    s1_drop_set  = set(s1_drop_idx.tolist())
    s1t_drop_set = set(s1t_drop_idx.tolist())
    # Indices that survive in each panel
    s1_keep_idx  = np.array([i for i in range(n) if i not in s1_drop_set],
                             dtype=np.int32)
    s1t_keep_idx = np.array([i for i in range(n) if i not in s1t_drop_set],
                             dtype=np.int32)

    # Evaluable cells:
    # cell i in s1 is evaluable iff:
    #   (1) it was NOT dropped from s1   → i in s1_keep_idx
    #   (2) its true match (also index i, identity map) was NOT dropped from s1_t
    s1t_keep_set = set(s1t_keep_idx.tolist())


    #only evaluate cells that are not dropped in s1 or s1_t
    eval_mask    = np.array([i not in s1t_drop_set
                             for i in s1_keep_idx], dtype=bool)

    
    eval_idx     = s1_keep_idx[eval_mask]   # indices into ORIGINAL panel1

    print(f"\n  Dropout applied  (mode={dropout_mode}, frac={dropout_frac:.0%})")
    print(f"    Original n           : {n}")
    print(f"    s1  kept / dropped   : {len(s1_keep_idx)} / {len(s1_drop_idx)}")
    print(f"    s1t kept / dropped   : {len(s1t_keep_idx)} / {len(s1t_drop_idx)}")
    print(f"    Evaluable cells      : {len(eval_idx)}  "
          f"({len(eval_idx)/n:.1%} of original)")
    print(f"    Non-evaluable (match missing): "
          f"{len(s1_keep_idx) - len(eval_idx)}")

    return {
        's1_keep_idx'  : s1_keep_idx,
        's1t_keep_idx' : s1t_keep_idx,
        'eval_idx'     : eval_idx,
        's1_drop_idx'  : s1_drop_idx,
        's1t_drop_idx' : s1t_drop_idx,
        'dropout_frac' : dropout_frac,
        'dropout_mode' : dropout_mode,
        'n_original'   : n,
        'n_s1_keep'    : len(s1_keep_idx),
        'n_s1t_keep'   : len(s1t_keep_idx),
        'n_eval'       : len(eval_idx),
    }


def evaluate_synthetic_with_dropout(
    s_orig,
    fake,
    dropout_result,
    morph_orig,
    morph_t,
    expr_orig,
    expr_t,
    MAX_DIST     = None,
    k_candidates = 20,
    pair_label="",
    spatial_key="spatial",
    threshold=None,
    extracted = True,
    N_PCA_MORPH = 256,
    n_gene_comps = 128
):
    """
    Evaluate matching strategies on a synthetic pair with cell dropout.

    How dropout affects matching and evaluation:
    ─────────────────────────────────────────────
    Matching runs on:
        Source (s1):    s1_keep_idx cells only    ← dropped s1 cells hidden
        Target (s1_t):  s1t_keep_idx cells only   ← dropped s1t cells hidden

    Accuracy evaluated on:
        eval_idx only  ← s1 cells whose true match still exists in s1_t
        (cells in s1_keep but not eval_idx have their match missing →
         they will match to wrong cells → excluded from accuracy)

    This means the algorithm:
        Can find correct matches for eval_idx cells
        Cannot find correct match for (s1_keep - eval_idx) cells
           → these are correctly excluded from accuracy computation

    Accuracy metrics:
        accuracy_matched : correct / n_matched_among_eval_idx
        accuracy_eval    : correct / len(eval_idx)
        coverage_eval    : n_matched_among_eval_idx / len(eval_idx)
    """
    # ── Slice to surviving cells ───────────────────────────────────
    s1_keep_idx  = dropout_result['s1_keep_idx']
    s1t_keep_idx = dropout_result['s1t_keep_idx']
    eval_idx     = dropout_result['eval_idx']   # original indices, subset of s1_keep


    # Coordinates
    xy_orig_full = _xy(s_orig, spatial_key)
    xy_orig_dropped      = xy_orig_full[s1_keep_idx]     # (n_s1_keep, 2)



    xy_t_full    = fake['xy']
    xy_t_dropped         = xy_t_full[s1t_keep_idx]       # (n_s1t_keep, 2)

    # Features — slice to kept cells
    morph_orig_dropped   = morph_orig[s1_keep_idx]               # (n_s1_keep, d)
    morph_t_dropped   = morph_t[s1t_keep_idx]                 # (n_s1t_keep, d)

    expr_orig_dropped   = expr_orig[s1_keep_idx]                # (n_s1_keep, d_expr)
    expr_t_dropped   = expr_t[s1t_keep_idx]                  # (n_s1t_keep, d_expr)

    n_query = len(xy_orig_dropped)

    # ── Run matching ───────────────────────────────────────────────
    # Matching sees: s1_keep cells vs s1t_keep cells
    # It does NOT know which cells were dropped
    match_results = run_all_strategies(
        xy_panel1    = xy_orig_dropped,
        xy_panel2_reg = xy_t_dropped,
        morph_panel1  = morph_orig_dropped,
        morph_panel2  = morph_t_dropped,
        expr_panel1   = expr_orig_dropped,
        expr_panel2   = expr_t_dropped,
        MAX_DIST      = MAX_DIST,
        k_candidates  = k_candidates,
        pair_label    = pair_label,
        threshold=threshold,
        extracted = extracted,
        N_PCA_MORPH = N_PCA_MORPH,
        n_gene_comps = n_gene_comps
    )

    results = {}
    n_eval = len(eval_idx)

    print(f"\n  {'Strategy':<20} {'Matched':>8} {'Eval_matched':>13} "
          f"{'Correct':>8} {'Acc_match':>10} {'Acc_eval':>10} {'Cov_eval':>10}")
    print(f"  {'-' * 83}")

    for strategy, (idx_p1_keep, idx_p2_keep) in match_results.items():

        if len(idx_p1_keep) == 0:
            results[strategy] = {
                "accuracy_matched" : 0.0,
                "accuracy_eval"    : 0.0,
                "coverage_eval"    : 0.0,
                "n_matched"        : 0,
                "n_eval_matched"   : 0,
                "n_correct"        : 0,
            }
            print(f"  {strategy:<20} {0:>8} {0:>13} {0:>8} "
                  f"{0:>10.3f} {0:>10.3f} {0:>10.3f}")
            continue

        # idx_p1_keep: positions within s1_keep
        # idx_p2_keep: positions within s1t_keep
        # Convert idx_p2_keep → original indices in fake['xy']
        original_p1_idx  = s1_keep_idx[idx_p1_keep]   # original s1 indices
        original_p2_idx  = s1t_keep_idx[idx_p2_keep]  # original s1t indices

        # Build lookup: original_s1_idx → matched_original_s1t_idx
        match_map = dict(zip(original_p1_idx.tolist(),
                             original_p2_idx.tolist()))

        # ── Evaluate ONLY on eval_idx ──────────────────────────────
        # eval_idx: original s1 cells whose true match exists in s1_t
        # ground truth: cell i maps to cell i (identity)
        eval_matched_mask = np.array([i in match_map for i in eval_idx],
                                      dtype=bool)
        eval_matched_p1   = eval_idx[eval_matched_mask]
        eval_matched_p2   = np.array([match_map[i]
                                       for i in eval_matched_p1.tolist()])

        n_eval_matched = eval_matched_mask.sum()
        correct        = (eval_matched_p2 == eval_matched_p1).sum()

        accuracy_matched = float(correct / n_eval_matched) \
                           if n_eval_matched > 0 else 0.0
        accuracy_eval    = float(correct / n_eval)
        coverage_eval    = float(n_eval_matched / n_eval)

        results[strategy] = {
            "accuracy_matched" : accuracy_matched,
            "accuracy_eval"    : accuracy_eval,
            "coverage_eval"    : coverage_eval,
            "n_matched"        : int(len(idx_p1_keep)),
            "n_eval_matched"   : int(n_eval_matched),
            "n_correct"        : int(correct),
            "n_eval"           : n_eval,
            "n_query"          : n_query,
        }

        print(f"  {strategy:<20} "
              f"{len(idx_p1_keep):>8} "
              f"{n_eval_matched:>13} "
              f"{correct:>8} "
              f"{accuracy_matched:>10.3f} "
              f"{accuracy_eval:>10.3f} "
              f"{coverage_eval:>10.3f}")

    return results


def run_synthetic_evaluation_with_dropout(
    adata_panel1,
    uni_panel1_normed,
    adata_panel2,
    uni_panel2_normed,
    expr_panel1,
    expr_pca,
    overlap_genes,
    spatial_key_p1    = "spatial",
    spatial_key_p2    = "spatial_registered",
    alpha_list        = [0.25, 0.5, 0.75, 1.0],
    n_seeds_per_alpha = 3,
    noise_scale       = 1.0,
    MAX_DIST          = 50.0,
    k_candidates      = 20,
    SAMPLE_NAME       = "s1",
    RECENTER          = False,
    NORMALIZE_EXPR    = False,
    N_PCA_MORPH       = 256,
    n_gene_comps = 128,
    xy_method         = "displacement_field",
    xy_k              = 20,
    xy_local_noise_weight = 0.25,
    dropout_frac      = 0.10,
    dropout_mode      = "independent",
    
    
    

    threshold=None,
    
    evaluate_hinge = False
):
    """
    Synthetic evaluation with cell dropout.

    Dropout is applied after creating the synthetic sample.
    Matching only sees the surviving cells.
    Accuracy is computed only on eval_idx (cells whose true match survived).

    Returns DataFrame with columns:
        sample, alpha, replicate, strategy, dropout_mode,
        accuracy_matched, accuracy_eval, coverage_eval,
        n_matched, n_eval_matched, n_correct, n_eval
    """

    # Step 1: Stats
    stats = compute_population_diff_stats(
        adata_panel1      = adata_panel1,
        uni_panel1_normed = uni_panel1_normed,
        adata_panel2      = adata_panel2,
        uni_panel2_normed = uni_panel2_normed,
        overlap_genes     = overlap_genes,
        spatial_key_p1    = spatial_key_p1,
        spatial_key_p2    = spatial_key_p2,
    )

    # Step 2: Synthetic replicates
    fakes = create_synthetic_replicates(
        adata_panel1          = adata_panel1,
        uni_panel1_normed     = uni_panel1_normed,
        stats                 = stats,
        alpha_list            = alpha_list,
        n_seeds_per_alpha     = n_seeds_per_alpha,
        noise_scale           = noise_scale,
        spatial_key           = spatial_key_p1,
        RECENTER              = RECENTER,
        xy_method             = xy_method,
        xy_k                  = xy_k,
        xy_local_noise_weight = xy_local_noise_weight,
    )

    strategies = ["A_distance", "B_histology", "C_expression", "D_combined"]
    records    = []

    for fake in fakes:
        rep   = fake['replicate']
        alpha = fake['alpha']

        print(f"\n{'='*60}")
        print(f"  {SAMPLE_NAME} → {SAMPLE_NAME}_t  "
              f"[alpha={alpha:.2f}  rep={rep+1}  "
              f"dropout={dropout_frac:.0%} ({dropout_mode})]")
        print(f"{'='*60}")

        # Step 3: Apply dropout
        dropout_result = apply_dropout(
            adata_panel1  = adata_panel1,
            dropout_frac  = dropout_frac,
            dropout_mode  = dropout_mode,
            random_seed   = rep * 777 + 42,  # different seed per replicate
        )
        
        if evaluate_hinge:
            eval_results = evaluate_synthetic(
                s_orig         = adata_panel1,
                s_t            = fake,
                
                morph_orig     = uni_panel1_normed,
                morph_t        = uni_panel2_normed,
                
                expr_orig      = expr_panel1,
                expr_t         = fake["adata"],
                
                MAX_DIST       = MAX_DIST,
                k_candidates   = k_candidates,
                pair_label     = f"{SAMPLE_NAME} α={alpha:.2f} rep={rep+1}",
                spatial_key    = spatial_key_p1,
                
                threshold      = threshold,
                extracted      = True,
                N_PCA_MORPH    = N_PCA_MORPH,
                n_gene_comps   = n_gene_comps,
            )   
        
        elif evaluate_hinge is False:
            # Step 4: Prepare features
            morph_list_orig, morph_list_fake, _ = prepare_morphology_features(
                [uni_panel1_normed], [fake["uni"]], N_PCA_MORPH=N_PCA_MORPH
            )
            morph_orig = morph_list_orig[0]   # (n_full, d)
            morph_t    = morph_list_fake[0]   # (n_full, d)

            
            
            expr_extracted_p1_list, expr_extracted_p2_list, _, overlap_genes = prepare_expression_features(
            [expr_panel1],[fake["adata"]], n_comps=n_gene_comps)
                        

            # Step 5: Evaluate with dropout
            eval_results = evaluate_synthetic_with_dropout(
                s_orig         = adata_panel1,
                fake           = fake,
                dropout_result = dropout_result,
                morph_orig     = morph_orig,
                morph_t        = morph_t,
                expr_orig      = expr_extracted_p1_list[0],
                expr_t         = expr_extracted_p2_list[0],
                MAX_DIST       = MAX_DIST,
                k_candidates   = k_candidates,
                pair_label     = f"{SAMPLE_NAME} α={alpha:.2f} rep={rep+1}",
                spatial_key    = spatial_key_p1,
                threshold      = threshold,
                extracted      = True,
                N_PCA_MORPH    = None, #no need to set this parameter since feature has been extracted
                n_gene_comps   =  None #no need to set this parameter since feature has been extracted
            )
            

        for strategy in strategies:
            rec = eval_results[strategy]
            records.append({
                "sample"           : SAMPLE_NAME,
                "alpha"            : alpha,
                "replicate"        : rep + 1,
                "strategy"         : strategy,
                "dropout_mode"     : dropout_mode,
                "dropout_frac"     : dropout_frac,
                "accuracy_matched" : rec["accuracy_matched"],
                "accuracy_eval"    : rec["accuracy_eval"],
                "coverage_eval"    : rec["coverage_eval"],
                "n_matched"        : rec["n_matched"],
                "n_eval_matched"   : rec["n_eval_matched"],
                "n_correct"        : rec["n_correct"],
                "n_eval"           : rec["n_eval"],
            })

    return pd.DataFrame(records), fakes, stats