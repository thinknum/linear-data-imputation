import numpy as np
import pandas as pd
import scipy.linalg


class Imputer():
    """
    This represents a model that can be used to do
    linear data imputation using its `impute` method.
    The `Imputer` class expects a pandas.DataFrame or
    an object that can be casted to a DataFrame.
    """
    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        self.names = df.columns
        df = df.astype('float64')
        self.mean = df.mean().values
        # this is 'better', but pd.DataFrame.cov is unbiased and
        # its results differ more from linear regression
        # cov = df.cov().values
        cov = self.nancov(df.values)
        cov[np.isnan(cov)] = 0
        self.inv_cov = self.generalized_inverse(cov)

    def impute(self, data):
        """
        This takes a single `data` argument, and returns
        the same `data` with all its missing values filled-in.
        `data` can be a:
        - dict
        - pandas.DataFrame
        - numpy.ndarray
        """
        data = data.copy()
        if isinstance(data, np.ndarray) and (len(data.shape) == 1):
            missing = np.isnan(data.astype(float))
            if not any(missing):
                return data
            k = (data[~missing]-self.mean[~missing]).astype(float)
            u = self.mean[missing]
            A, B = self.split_matrix(self.inv_cov[0], missing)
            s0 = np.linalg.lstsq(A, B.dot(k), rcond=None)[0]
            if self.inv_cov[1].any():
                C, D = self.split_matrix(self.inv_cov[1], missing)
                s1 = np.linalg.lstsq(C, D.dot(k), rcond=None)[0]
                ns = scipy.linalg.null_space(C)
                data[missing] = u + s1 + ns @ ns.T @ s0
            else:
                data[missing] = u + s0
            return data
        elif isinstance(data, np.ndarray):
            return np.array([self.impute(x) for x in data])
        elif isinstance(data, pd.DataFrame):
            data[self.names] = self.impute(data[self.names].values)
            return data
        elif isinstance(data, dict):
            data.update(zip(
                self.names,
                self.impute(np.array([data[n] for n in self.names]))
            ))
            return data

    @staticmethod
    def split_matrix(M, missing):
        A = M[np.ix_(missing, missing)]
        B1 = M[np.ix_(missing, ~missing)]
        B2 = M[np.ix_(~missing, missing)]
        return A+A.T, -(B1+B2.T)

    @staticmethod
    def nancov(matrix):
        matrix = matrix - np.nanmean(matrix, axis=0)
        return np.array([
            np.nanmean(matrix.T * column, axis=1)
            for column in matrix.T])

    @staticmethod
    def generalized_inverse(A):
        """
        This method is a generalization of the inverse matrix for singular matrices.
        It returns a tuple with 2 matrices. You can think of these 2 matrices as
        a single matrix with a 'real' part and an 'infinite' part.
        This matrix will have an 'infinite' part iff the input matrix is singular.
        The 'real' part is the Moore–Penrose inverse.
        """
        U, s, Vt = np.linalg.svd(A)
        rcond = np.finfo(s.dtype).eps * max(U.shape[0], Vt.shape[1])
        tol = np.amax(s) * rcond
        G1 = np.divide(1, s, out=np.zeros_like(s), where=(s > tol))
        G2 = (s <= tol).astype(float)
        return (U @ np.diag(G1) @ Vt).T, (U @ np.diag(G2) @ Vt).T


def impute(data):
    return Imputer(data).impute(data)
