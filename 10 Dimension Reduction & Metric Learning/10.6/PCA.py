import numpy as np

class PCA(object):
    """ an implementation of PCA (Principal Component Analysis) """

    def __init__(self):
        pass

    def solve(self, xs, ndim):
        """ reduce xs' dimensionality to ndim """
        for idx in range(xs.shape[0]): xs[idx] -= np.mean(xs[idx])      # zero centralized
        eigen_vals, eigen_vecs = np.linalg.eig(np.matmul(xs.T, xs))     # eigenvalue decomposition
        eigen_sort_idx = np.argsort(-eigen_vals)                        # sort eigenvalues (descending order)
        selected_eigen_vecs = eigen_vecs.T[eigen_sort_idx][:ndim]       # select eigenvectors with Top-ndim eigenvalues
        return selected_eigen_vecs                                      # mapping matrix