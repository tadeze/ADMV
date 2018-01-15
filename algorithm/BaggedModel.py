import numpy as np

class BaggedModel(object):
    def __init__(self, model=None, num_model=100, nsample=100):
        self.models = []
        self.num_model = num_model
        self.model_features = np.nan
        self.model = model
        self.nsample = nsample
        if model is None:
            raise ValueError("Basic model object is required.")
    def train(self, X):
        assert isinstance(X, np.ndarray)
        n, d  = X.shape
        self.model_features = np.zeros([self.num_model,d])
        d_subset = np.ceil(d/np.sqrt(d))

        for model in range(self.num_model):
            sample_index = np.random.choice(n, self.nsample, False)
            model_l = self.model()




    # def score(self, X, check_miss=True):
    #     pass
    #
    #
    # self.trees_proj = None
    #     self.num_tree_used = []
    #     self.check_miss = False
    #     self.max_height = max_height
    #     self.trees = []
    # def train(self, train_df):
    #     assert isinstance(train_df, np.ndarray)
    #     nrow, ncol = train_df.shape
    #     self.trees_proj = np.zeros([self.ntree, ncol])
    #     if self.nsample > nrow:
    #         self.nsample = nrow
    #     n_bagged = int(np.ceil(ncol/np.sqrt(ncol)))
    #     for tree in range(self.ntree):
    #         # generate rotation matrix
    #         sample_index = np.random.choice(nrow, self.nsample, False)
    #         itree = pft.IsolationTree()
    #         #itree.train_points = sample_index
    #         cols = np.random.choice(ncol, n_bagged, False)
    #         #print cols
    #         itree.iTree(sample_index, train_df[:,cols], 0, self.max_height)
    #         self.trees.append({"tree": itree, "cols":cols})
    #         self.trees_proj[tree,cols] = 1
    #
    #         #logger.info("tree %d, %s"%(tree,cols))
    # def score(self, test_df, check_miss=True, faster=False):
    #     self.num_tree_used = []# np.zeros([test_df.shape[0],1])
    #     self.check_miss = check_miss
    #     self.faster = faster
    #     return super(BaggedIForest, self).score(test_df)
