from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils.validation import check_array, check_is_fitted
from util.common import *
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
    def __init__(self, num_model=10):
        self.num_model = num_model
        self.model_features = None
        self.models = []
    def train(self, train_df):
        assert isinstance(train_df, np.ndarray)
        nrow, ncol = train_df.shape
        self.model_features = np.zeros([self.num_model, ncol])
        #if self.nsample > nrow:
        #    self.nsample = nrow
        n_bagged = int(np.ceil(ncol/np.sqrt(ncol)))
        for lof_model in range(self.num_model):
            #sample_index = np.random.choice(nrow, self.nsample, False)
            lof = LOF(n_neighbors=20)
            cols = np.random.choice(ncol, n_bagged, False)
            lof = lof.train(train_df[:,cols])
            #itree.iTree(sample_index, train_df[:,cols], 0, self.max_height)
            self.models.append({"lof": lof, "cols":cols})
            self.model_features[lof_model,cols] = 1

    def score(self, test_df, check_miss=True):
        self.num_models_used = []
        self.check_miss = check_miss
#        return super(BaggedIForest, self).score(test_df)

        if self.check_miss:
            miss_column = get_miss_features(test_df)
        else:
            miss_column = []
        all_scores = []
        non_missing_models = available_models(self.model_features, miss_column)
        maskcol = np.ones(test_df.shape, dtype=bool)
        maskcol[miss_column] = False
        if len(non_missing_models) < 1:
            return 0.0

        for model_inst in non_missing_models:
            lof = self.models[model_inst]
            sliced_data = test_df[lof["cols"]]
            all_scores.append(lof["lof"].score(sliced_data))
        self.num_models_used.append(len(non_missing_models))
        return np.mean(all_scores)


if __name__ == '__main__':

    ll = LOF()
    w = np.random.randn(100,5)
    w[1,2] = 20
    w[5,0]  = -30
    ff = ll.train(w)
    print ff.score(w)
    #import sklearn
    #print sklearn.__version__
    lb = BaggedLOF()
    lb.train(w)
    #print lb.models
    lb.score(w, False)