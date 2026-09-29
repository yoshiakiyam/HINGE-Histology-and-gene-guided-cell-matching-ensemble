
import numpy as np

from scipy.sparse import issparse
from sklearn.preprocessing import normalize
import scanpy as sc



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