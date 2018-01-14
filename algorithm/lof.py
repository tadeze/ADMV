from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils.validation import check_array, check_is_fitted
import numpy as np

class LOF(LocalOutlierFactor):


    def __init__(self, n_neighbors=20):
        super(LOF, self).__init__(n_neighbors=n_neighbors, algorithm='auto', leaf_size=30,
                 metric='minkowski', p=2, metric_params=None,
                 contamination=0.1, n_jobs=1)

    def train(self, X):
        return self.fit(X)

    def score(self, X=None):

        check_is_fitted(self, ["threshold_", "negative_outlier_factor_",
                               "n_neighbors_", "_distances_fit_X_"])

        if X is not None:
            X = check_array(X, accept_sparse='csr')
            is_inlier = np.ones(X.shape[0], dtype=int)
            is_inlier[self._decision_function(X) <= self.threshold_] = -1
            is_inlier = self._decision_function(X)
        else:
            is_inlier = np.ones(self._fit_X.shape[0], dtype=int)
            is_inlier[self.negative_outlier_factor_ <= self.threshold_] = -1
            is_inlier = self.negative_outlier_factor_
        return is_inlier

class BaggedLOF(object):
    def __init__(self):
        pass


if __name__ == '__main__':

    ll = LOF()
    w = np.random.randn(100,3)
    w[1,2] = 20
    w[5,0]  = -30
    ff = ll.train(w)
    print ff.score(w)
    #import sklearn
    #print sklearn.__version__