import algorithm.pyad as pft
import numpy as np
#import time
NA = np.nan #-9999.0

class BaggedIForest(pft.IForest):
    def __init__(self, ntree=100, nsample=512,
                 max_height=0, adaptive=False, rangecheck=True):
        super(BaggedIForest,self).__init__(train_df=None, ntree=ntree, nsample=nsample,
                                  max_height=max_height, adaptive=adaptive,rangecheck=rangecheck)
        self.trees_proj = None
        self.num_tree_used = []
        self.check_miss = False
        self.max_height = max_height
        self.trees = []
    def train(self, train_df):
        assert isinstance(train_df, np.ndarray)
        nrow, ncol = train_df.shape
        self.trees_proj = np.zeros([self.ntree, ncol])
        if self.nsample > nrow:
            self.nsample = nrow
        n_bagged = int(np.ceil(ncol/np.sqrt(ncol)))
        for tree in range(self.ntree):
            sample_index = np.random.choice(nrow, self.nsample, False)
            itree = pft.IsolationTree()
            cols = np.random.choice(ncol, n_bagged, False)
            itree.iTree(sample_index, train_df[:,cols], 0, self.max_height)
            self.trees.append({"tree": itree, "cols":cols})
            self.trees_proj[tree,cols] = 1

    def score(self, test_df, check_miss=True):
        self.num_tree_used = []# np.zeros([test_df.shape[0],1])
        self.check_miss = check_miss
        return super(BaggedIForest, self).score(test_df)

    def get_trees(self, miss_features):
        # Get trees without missing features.
        miss_trees = self.trees_proj[:,miss_features]
        used_trees = np.where(self.trees_proj.any(axis=1) != NA)
        return used_trees

    def get_miss_features(self, row):
        """
        :param row: np.ndarray of 1Xd vector.
        :return:
        """
        if np.isnan(NA):
            miss_column = np.where(np.isnan(row))[0]
        else:
            miss_column = np.where(row == NA)[0]
        return miss_column
    def available_trees(self, miss_column):
        """
        Return trees without the missing column
        :param miss_column:
        :return:
        """
        if len(miss_column)==0:
            return range(0,self.trees_proj.shape[0])
        with_miss_column = self.trees_proj[:,miss_column]
        available_trees = np.where(~with_miss_column.any(axis=1))[0]
        return available_trees
    def depth(self, test_df, oob=False):
        if self.check_miss:
            miss_column = self.get_miss_features(test_df)
        else:
            miss_column = []
        all_depth =[]
        non_missing_trees = self.available_trees(miss_column)
        maskcol = np.ones(test_df.shape, dtype=bool)
        maskcol[miss_column] = False
        if len(non_missing_trees)<1:
            return 0.0
        for trees_inst in non_missing_trees:
            tree = self.trees[trees_inst]
            sliced_data = test_df[tree["cols"]]
            all_depth.append(tree["tree"].path_length(sliced_data))
        self.num_tree_used.append(len(non_missing_trees))
        return all_depth


if __name__ == '__main__':
#<type 'list'>: [14.0, 14.0, 5.0, 13.0, 17.0, 12.0, 17.0]
    w = np.random.randn(20,5)
    test = w.copy()

    # ff = pft.BaggedIForest(ntree=100)
    # ff.train(w)
    test[1:5,[2,3]] = NA
    test[8,4] = NA
    # to = time.time()
    # ff.score(test )[1:10], #ff.num_tree_used
    # #ff.score(test)[1:10], #ff.num_tree_used
    # print time.time() - to
    # print "Done"

    # ff = BaggedIForest(ntree=1000)
    # ff.train(w)
    # test[1:5, [2, 3, 5]] = NA
    # test[0, 4] = NA
    # #np.random.seed(90)
    # to = time.time()
    # print ff.score(test, check_miss=True)[0:10], ff.num_tree_used
    # #ff.score(test, check_miss=False)[1:10], ff.num_tree_used
    # print time.time() - to
    #
    # to = time.time()
    # print ff.score(test, check_miss=True)[0:10], ff.num_tree_used
    # print time.time() - to

## The benchmark is the same, may be use different
    # print ff.nsample
    # print ff.depth(w[1,:])
    # print test[0:7,:]
    #
    # import fancyimpute as fi
    # test_imp = fi.KNN(k=3).complete(test)
    # print ff.score(test_imp, check_miss=True)[1:10], ff.num_tree_used
    import loda.loda as ld
    ldd = ld.Loda(maxk=100)
    ldd.train(w)
    print test[1:5,:]
    test[2,3] = -50
    test[3,0] = 20
    print ldd.score(test, True)
    print ldd.score(test, False)

    #print ff.num_tree_used

    ff = pft.IsolationForest()



    # def train_forest(train_x, check_mv=False, tree_size=1):
#     ff = pft.IsolationForest(train_x, ntree=100 * tree_size, nsample=512)
#     return ff
#
# def score(test_x, cmv=False):
#     sc_gt = ff.score(test_x, cmv)
#     return sc_gt