
import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.sparse import issparse
import scanpy as sc






"""
    PCA-compress and L2-normalize UNI embeddings for each pair.

    Fits one PCA jointly on ALL samples (480 + 5k combined) so both
    panels live in the same morphological feature space.

    Returns two lists of L2-normalized arrays aligned to `pairs`.
"""
def prepare_morphology_features(uni_panel1_list, uni_panel2_list,
                                  N_PCA_MORPH=100, verbose = False):

    if verbose:
        print("=" * 55)
        print("  Preparing morphology features (UNI embeddings)")
        print("=" * 55)

        print("combining UNI features from both panels...")
    all_X = []
    # Fit PCA on all samples combined
    for adata in uni_panel1_list + uni_panel2_list:
        X = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)
        all_X.append(X.astype(np.float32))
    all_X = np.vstack(all_X)



    n_comps_actual = min(N_PCA_MORPH, all_X.shape[1] - 1) #get the number of components, either N_PCA_MORPH or the number of features - 1, whichever is smaller
    
    # Fit PCA on all samples combined
    if verbose:
        print(f"  Fitting PCA on {all_X.shape} UNI features "
              f"→ {n_comps_actual} components...")
    pca = PCA(n_components=n_comps_actual, random_state=42)
    pca.fit(all_X)
    if verbose:
        print(f"  Variance retained: {pca.explained_variance_ratio_.sum():.1%}")
    del all_X

    morph_panel1_list = []
    morph_panel2_list = []
    
    
    # Transform and normalize each panel's UNI features
    for uni_panel1, uni_panel2 in zip(uni_panel1_list, uni_panel2_list):
        X_panel1 = uni_panel1.X
        X_panel2 = uni_panel2.X
        if issparse(X_panel1): X_panel1 = X_panel1.toarray()
        if issparse(X_panel2):  X_panel2  = X_panel2.toarray()

        m_panel1 = normalize(pca.transform(X_panel1.astype(np.float32)))
        m_panel2  = normalize(pca.transform(X_panel2.astype(np.float32)))

        morph_panel1_list.append(m_panel1)
        morph_panel2_list.append(m_panel2)
        if verbose:
            print( f"panel 1 morph {m_panel1.shape}, panel 2 morph {m_panel2.shape}")

    return morph_panel1_list, morph_panel2_list, pca

#normalize and log transform expression data 
def get_normalized_X(adata, genes):
        a = adata[:, genes].copy()
        sc.pp.normalize_total(a)
        sc.pp.log1p(a)
        X = a.X.toarray() if issparse(a.X) else np.array(a.X)
        return X
    

from HINGE.utils import get_overlapping_genes
def prepare_expression_features(rna_panel1, rna_panel2, n_comps=50, normalize_expr = False, verbose = False):
    
    
    
    overlap_genes = get_overlapping_genes(rna_panel1, rna_panel2)
    
    
    if verbose:
        print("\n" + "=" * 55)
        print("  Preparing expression features (overlapping genes PCA)")
        print("=" * 55)
    
        print(f"  Overlapping genes across all samples: {len(overlap_genes)}")

    

    
    if normalize_expr:
        
        X_panel1_all = np.vstack([
            get_normalized_X(rna_panel1_sample, overlap_genes)
            for rna_panel1_sample in rna_panel1
        ])
        
    else:
        # :white_check_mark: Extract .X and convert to dense numpy array
        X_panel1_all = np.vstack([
            rna_panel1_sample[:, overlap_genes].X.toarray()
            if issparse(rna_panel1_sample[:, overlap_genes].X)
            else np.array(rna_panel1_sample[:, overlap_genes].X)
            for rna_panel1_sample in rna_panel1
        ])
    # Fit PCA on all of panel 1 samples combined
    n_comps_actual = min(n_comps, len(overlap_genes) - 1)
    
    if verbose:
    
        print(f"  Fitting expression PCA on {X_panel1_all.shape} "
            f"→ {n_comps_actual} components...")
    pca = PCA(n_components=n_comps_actual, random_state=42)
    pca.fit(X_panel1_all)
    if verbose:
        print(f"  Variance retained: {pca.explained_variance_ratio_.sum():.1%}")
    del X_panel1_all

    expr_panel1_list = []
    expr_panel2_list  = []

    for rna_panel1_sample, rna_panel2_sample in zip(rna_panel1, rna_panel2):
        
        if normalize_expr:
            X_panel1 = get_normalized_X(rna_panel1_sample, overlap_genes)
            X_panel2  = get_normalized_X(rna_panel2_sample,  overlap_genes)
        else:
            X_panel1 = (rna_panel1_sample[:, overlap_genes].X.toarray()
                        if issparse(rna_panel1_sample[:, overlap_genes].X)
                        else np.array(rna_panel1_sample[:, overlap_genes].X))
        
            X_panel2  = (rna_panel2_sample[:, overlap_genes].X.toarray()
                         if issparse(rna_panel2_sample[:, overlap_genes].X)
                         else np.array(rna_panel2_sample[:, overlap_genes].X))
        

        # Transform and normalize each panel's expression
        e_panel1 = normalize(pca.transform(X_panel1))
        e_panel2  = normalize(pca.transform(X_panel2))

        expr_panel1_list.append(e_panel1)
        expr_panel2_list.append(e_panel2)

        if verbose:
            print(f"Panel 1 expr {e_panel1.shape}, Panel 2 expr {e_panel1.shape}")


    return expr_panel1_list, expr_panel2_list, pca, overlap_genes









def harmonize_uni_panels(
    uni_panel1_normed,
    uni_panel2_normed,
    method          = "mean_center",  # "mean_center" | "pca_mean_center" | "harmony"
    n_pca_comps     = 50,             # used by pca_mean_center and harmony
    n_harmony_iters = 10,
    harmony_theta   = 2.0,
    renormalize     = True,           # L2-renormalize after correction
    verbose         = True,
):
    """
    Preprocessing: remove inter-panel mean shift from L2-normalized UNI embeddings.

    Call this ONCE before run_synthetic_evaluation_with_dropout.
    The corrected embeddings replace the raw ones for all strategies (A–J).

    Why at this level (not inside HINGE6 per replicate)
    ----------------------------------------------------
    - Paid once, not per replicate (12x speedup for 12-replicate eval)
    - All strategies (A, B, C, D, E, I …) see the same corrected features
    - The real inter-panel UNI shift is a fixed property of the scanner /
      tissue section pair — it should be corrected globally, not per match call
    - Synthetic P2 is created from P1 after harmonization, so the synthetic
      shift (alpha-scaled noise) is added on top of already-aligned features

    Methods
    -------
    "mean_center"     : subtract per-panel mean in 2048-dim feature space,
                        re-add global mean. No extra dependencies.
    "pca_mean_center" : joint PCA → subtract per-panel mean in PCA space →
                        project back to feature space. More targeted.
    "harmony"         : joint PCA → Harmony batch correction → project back.
                        Requires harmonypy. Strongest correction.

    Parameters
    ----------
    renormalize : bool
        L2-renormalize each cell after correction so cosine similarities
        remain valid. Should almost always be True.

    Returns
    -------
    uni_p1_harmonized : AnnData  (same obs/var as input, corrected .X)
    uni_p2_harmonized : AnnData  (same obs/var as input, corrected .X)
    harmony_info      : dict     (diagnostics)
    """
    from scipy.sparse import issparse
    from sklearn.preprocessing import normalize as sk_normalize
    import anndata as ad
    import scanpy as sc

    def _to_array(x):
        if hasattr(x, 'X'):
            arr = x.X.toarray() if issparse(x.X) else np.array(x.X)
        else:
            arr = np.array(x)
        return arr.astype(np.float32)

    def _to_adata(original, X_new):
        """Return AnnData with corrected .X, preserving obs/var/obsm."""
        adata_out = ad.AnnData(X=X_new.copy())
        if hasattr(original, 'obs'):
            adata_out.obs    = original.obs.copy()
        if hasattr(original, 'var'):
            adata_out.var    = original.var.copy()
        if hasattr(original, 'obsm'):
            adata_out.obsm   = {k: v.copy() for k, v in original.obsm.items()
                                if k != 'X_pca'}  # drop stale PCA
        return adata_out

    X_p1 = _to_array(uni_panel1_normed)   # (n1, D)
    X_p2 = _to_array(uni_panel2_normed)   # (n2, D)
    n1, n2, D = len(X_p1), len(X_p2), X_p1.shape[1]

    if np.isnan(X_p1).any() or np.isnan(X_p2).any():
        raise ValueError("harmonize_uni_panels: NaN detected in input features.")

    # ── Pre-harmonization mixing diagnostic ──────────────────────────────────
    def _mixing_std_knn(X, n1_cells, k=15):
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)
        nn.fit(X)
        _, indices = nn.kneighbors(X)
        return float(np.std([np.mean(idx < n1_cells) for idx in indices]))

    if verbose:
        # Use PCA-compressed version for diagnostic speed
        X_joint_raw = np.vstack([X_p1, X_p2])
        adata_diag  = sc.AnnData(X=X_joint_raw)
        n_safe      = min(n_pca_comps, D - 1, len(X_joint_raw) - 1)
        sc.pp.pca(adata_diag, n_comps=n_safe)
        pca_raw     = np.array(adata_diag.obsm['X_pca'])
        std_before  = _mixing_std_knn(pca_raw, n1)
        expected    = n1 / (n1 + n2)
        print(f"\n  Panel harmonization  (method={method})")
        print(f"    n1={n1}, n2={n2}, D={D}")
        print(f"    Expected P1 frac           : {expected:.3f}")
        print(f"    Mixing std BEFORE harmony  : {std_before:.4f}")

    # ── Apply correction ──────────────────────────────────────────────────────
    if method == "mean_center":
        # Subtract per-panel mean, re-add global mean (in full feature space)
        mean_p1     = X_p1.mean(axis=0)                         # (D,)
        mean_p2     = X_p2.mean(axis=0)                         # (D,)
        global_mean = (n1 * mean_p1 + n2 * mean_p2) / (n1 + n2)

        shift_p1_mag = float(np.linalg.norm(mean_p1 - global_mean))
        shift_p2_mag = float(np.linalg.norm(mean_p2 - global_mean))

        X_p1_corr = X_p1 - (mean_p1 - global_mean)
        X_p2_corr = X_p2 - (mean_p2 - global_mean)

        if verbose:
            print(f"    P1 centroid shift corrected : {shift_p1_mag:.4f}")
            print(f"    P2 centroid shift corrected : {shift_p2_mag:.4f}")

        harmony_info = {
            'method': 'mean_center',
            'shift_p1': shift_p1_mag,
            'shift_p2': shift_p2_mag,
        }

    elif method == "pca_mean_center":
        # Joint PCA → mean-center in PCA space → reconstruct feature space
        X_joint = np.vstack([X_p1, X_p2])
        adata_j = sc.AnnData(X=X_joint)
        n_safe  = min(n_pca_comps, D - 1, len(X_joint) - 1)
        sc.pp.pca(adata_j, n_comps=n_safe)

        pca_joint   = np.array(adata_j.obsm['X_pca'])          # (n1+n2, K)
        pca_p1      = pca_joint[:n1]
        pca_p2      = pca_joint[n1:]
        mean_pca_p1 = pca_p1.mean(axis=0)
        mean_pca_p2 = pca_p2.mean(axis=0)
        global_mean_pca = (n1 * mean_pca_p1 + n2 * mean_pca_p2) / (n1 + n2)

        pca_p1_corr = pca_p1 - (mean_pca_p1 - global_mean_pca)
        pca_p2_corr = pca_p2 - (mean_pca_p2 - global_mean_pca)

        # Project back to feature space using PCA components
        components  = adata_j.varm['PCs'].T                     # (K, D)
        X_p1_corr   = pca_p1_corr @ components + X_joint.mean(axis=0)
        X_p2_corr   = pca_p2_corr @ components + X_joint.mean(axis=0)

        shift_p1_mag = float(np.linalg.norm(mean_pca_p1 - global_mean_pca))
        shift_p2_mag = float(np.linalg.norm(mean_pca_p2 - global_mean_pca))

        if verbose:
            print(f"    PCA space P1 shift corrected: {shift_p1_mag:.4f}")
            print(f"    PCA space P2 shift corrected: {shift_p2_mag:.4f}")

        harmony_info = {
            'method': 'pca_mean_center',
            'shift_p1_pca': shift_p1_mag,
            'shift_p2_pca': shift_p2_mag,
            'n_pca_comps': n_safe,
        }

    elif method == "harmony":
        try:
            X_joint = np.vstack([X_p1, X_p2])
            adata_j = sc.AnnData(X=X_joint)
            adata_j.obs['panel'] = ['P1'] * n1 + ['P2'] * n2
            n_safe = min(n_pca_comps, D - 1, len(X_joint) - 1)
            sc.pp.pca(adata_j, n_comps=n_safe)
            sc.external.pp.harmony_integrate(
                adata_j,
                key            = 'panel',
                basis          = 'X_pca',
                adjusted_basis = 'X_pca_harmony',
                max_iter_harmony = n_harmony_iters,
                theta          = harmony_theta,
                random_state   = 42,
            )
            pca_harm    = np.array(adata_j.obsm['X_pca_harmony'])
            components  = adata_j.varm['PCs'].T
            X_joint_mean = X_joint.mean(axis=0)
            X_p1_corr   = pca_harm[:n1] @ components + X_joint_mean
            X_p2_corr   = pca_harm[n1:] @ components + X_joint_mean
            if verbose:
                print(f"    Harmony applied  "
                      f"(theta={harmony_theta}, iters={n_harmony_iters})")
            harmony_info = {
                'method': 'harmony',
                'theta': harmony_theta,
                'n_harmony_iters': n_harmony_iters,
            }
        except Exception as e:
            print(f"  ⚠️  Harmony failed ({e}), falling back to mean_center.")
            return harmonize_uni_panels(
                uni_panel1_normed, uni_panel2_normed,
                method='mean_center', renormalize=renormalize, verbose=verbose
            )

    elif method == "none":
        if verbose:
            print("    No harmonization applied.")
        harmony_info = {'method': 'none'}
        return uni_panel1_normed, uni_panel2_normed, harmony_info

    else:
        raise ValueError(f"Unknown method '{method}'. "
                         f"Choose: 'mean_center', 'pca_mean_center', 'harmony', 'none'")

    # ── L2 renormalization ────────────────────────────────────────────────────
    if renormalize:
        X_p1_corr = sk_normalize(X_p1_corr.astype(np.float64)).astype(np.float32)
        X_p2_corr = sk_normalize(X_p2_corr.astype(np.float64)).astype(np.float32)

    # ── Post-harmonization diagnostic ────────────────────────────────────────
    if verbose:
        X_joint_corr = np.vstack([X_p1_corr, X_p2_corr])
        adata_post   = sc.AnnData(X=X_joint_corr)
        n_safe2      = min(n_pca_comps, X_joint_corr.shape[1] - 1,
                           len(X_joint_corr) - 1)
        sc.pp.pca(adata_post, n_comps=n_safe2)
        pca_corr  = np.array(adata_post.obsm['X_pca'])
        std_after = _mixing_std_knn(pca_corr, n1)
        improvement = std_before - std_after
        flag = "✅" if improvement > 0.01 else "⚠️  minimal improvement"
        print(f"    Mixing std AFTER  harmony  : {std_after:.4f}")
        print(f"    Improvement                : {improvement:.4f}  {flag}")
        harmony_info['mixing_std_before'] = std_before
        harmony_info['mixing_std_after']  = std_after

    # ── Wrap back into AnnData ────────────────────────────────────────────────
    uni_p1_harmonized = _to_adata(uni_panel1_normed, X_p1_corr)
    uni_p2_harmonized = _to_adata(uni_panel2_normed, X_p2_corr)

    return uni_p1_harmonized, uni_p2_harmonized, harmony_info