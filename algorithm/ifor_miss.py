import pyad as pft
import numpy as np
## Build Forest using bagged features.

# class IForest(object):
#     """
#     IForest version using most python code. This is used for experimenting version.
#     """
#     def __init__(self, train_df=None, ntree=100, nsample=512,
#                  max_height=0, adaptive=False, rangecheck=True, check_missing_value=True):
#         self.nsample = nsample
#         self.ntree = ntree
#         self.rangecheck = rangecheck
#         self.adaptive = adaptive
#         self.maxheight = max_height
#         self.check_missing_value = check_missing_value
#         self.rot_trees = []
#         self.trees = []
#         #self.sparsity = sparsity
#         if train_df is not None:
#             self.train_df = train_df
#             self.train(self.train_df)
#
#     def train(self, train_df):
#         """ Train forest
#         :type train_df: numpy.ndarray training dataset
#         """
#
#         assert isinstance(train_df, np.ndarray)
#         nrow, ncol = train_df.shape
#         if self.nsample > nrow:
#             self.nsample = nrow
#         for tree in range(self.ntree):
#             # generate rotation matrix
#             sample_index = np.random.choice(nrow, self.nsample, False)
#             itree = pft.IsolationTree()
#             itree.train_points = sample_index
#             itree.iTree(sample_index, train_df, 0, self.maxheight)
#             self.trees.append({"tree": itree})
#
#     def depth(self, test_df, oob=False):
#
#         depth = []
#
#         for tree_inst in self.trees:
#             #rot_mat = rottree["rotmat"]
#             tree = tree_inst["tree"]
#             depth.append(tree.path_length(test_df))
#         return depth
#
#     def average_depth(self, test_df):
#         assert isinstance(test_df, np.ndarray)
#         avg_depth = [np.mean(self.depth(row)) for row in test_df]
#         return np.array(avg_depth)
#
#     def oob_depth(self, test_df):
#         """Compute depth using the out of bag trees"""
#         assert isinstance(test_df, np.ndarray)
#         depth = []
#         for rottree in self.trees:
#             tree = rottree["tree"]
#             if test_df not in tree.train_points:
#                 depth.append(tree.path_length(test_df))
#         return (len(depth), np.mean(depth))
#
#     def explanation(self, query_point):
#         #explanation=[]
#         features = collection.defaultdict(float)
#         for rottree in self.trees:
#             tree = rottree["tree"]
#             for index, depth_inv in tree.explanation(query_point).items():
#                 features[index] += depth_inv
#         return features
#
#     def score(self, test_df):
#         #score of allpoints
#         bst = lambda n: 0.0 if (n - 1) <= 0 else (2.0 * np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n
#         avg_depth = self.average_depth(test_df)
#         scores = np.power(2, (-1 * avg_depth / bst(self.nsample)))
#         return scores
import logging
logger = logging.getLogger(__name__)
NA = np.nan #-9999.0
class BaggedIForest(pft.IForest):
    def __init__(self, ntree=100, nsample=512,
                 max_height=0, adaptive=False, rangecheck=True):
        super(BaggedIForest,self).__init__(train_df=None, ntree=ntree, nsample=nsample,
                                  max_height=max_height, adaptive=adaptive,rangecheck=rangecheck)
        self.trees_proj = None
        self.num_tree_used = []
    def train(self, train_df):
        assert isinstance(train_df, np.ndarray)
        nrow, ncol = train_df.shape
        self.trees_proj = np.zeros([self.ntree, ncol])
        if self.nsample > nrow:
            self.nsample = nrow
        n_bagged = int(np.ceil(ncol/np.sqrt(ncol)))
        for tree in range(self.ntree):
            # generate rotation matrix
            sample_index = np.random.choice(nrow, self.nsample, False)
            itree = pft.IsolationTree()
            #itree.train_points = sample_index
            cols = np.random.choice(ncol, n_bagged, False)
            #print cols
            itree.iTree(sample_index, train_df[:,cols], 0, self.maxheight)
            self.trees.append({"tree": itree, "cols":cols})
            self.trees_proj[tree,cols] = 1
            #logger.info("tree %d, %s"%(tree,cols))
     #def score(self, test_df):
        #self.num_tree_used = np.zeros([test_df.shape[0],1])
     #   return super(BaggedIForest, self).score(test_df)

    def get_trees(self, miss_features):
        # Get trees without missing features.
        miss_trees = self.trees_proj[:,miss_features]
        used_trees = np.where(self.trees_proj.any(axis=1) != NA)
        return used_trees

    def get_miss_features(self, test_df):
        miss_column = np.where(test_df == NA)[0]
        return miss_column

    def depth(self, test_df, oob=False):
        """
        :param test_df: ndarray 1xD vector of datapoint
        :param oob:
        :return:list list of depth from trees.
        """
        all_depth = []
        miss_column = self.get_miss_features(test_df)
        for tree_inst in self.trees:
            tree = tree_inst["tree"]
            used_cols = tree_inst["cols"]
            if tree_inst["cols"].any():
                if np.intersect1d(miss_column, used_cols).any():
                    continue
                all_depth.append(tree.path_length(test_df[tree_inst["cols"]]))
            else:
                all_depth.append(tree.path_length(test_df))
        self.num_tree_used.append(len(all_depth))
        return all_depth

if __name__ == '__main__':
    ff = BaggedIForest(ntree=1000)
    w = np.random.randn(50,10)
    test = w.copy()
    ff.train(w)
    test[1:5,[2,3,5,6,8]] = NA
    test[8,4] = NA
    print ff.nsample
    print ff.depth(w[1,:])
    print ff.score(test)
    print ff.num_tree_used
#  Full forest
    #fx = BaggedIForest(ntree=10)
    #super(BaggedIForest, fx).train(w)
    #print fx.score(test)



    # def train_forest(train_x, check_mv=False, tree_size=1):
#     ff = pft.IsolationForest(train_x, ntree=100 * tree_size, nsample=512)
#     return ff
#
# def score(test_x, cmv=False):
#     sc_gt = ff.score(test_x, cmv)
#     return sc_gt