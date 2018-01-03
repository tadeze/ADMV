import pyad as pft

## Build Forest using bagged features.

class iFor(object):
    def __init__(self):
        self.ntree = None
    def train(self, train_x):
        nrow, ncol = train_x.shape
        forest = {}
        for ntree in self.ntree:
            features = feature_bag(ncol)
            tree = pft.Tree(train_x[,features])
            forest.append({tree:tree, features: features})
        self.forest = forest
    def score(self, test_x):
        """
        Build the depth using the trained forest.
        :param test_x:
        :return:
        """
        for tree, features in self.forest:
            depth


def train_forest(train_x, check_mv=False, tree_size=1):
    ff = pft.IsolationForest(train_x, ntree=100 * tree_size, nsample=512)
    return ff

def score(test_x, cmv=False):
    sc_gt = ff.score(test_x, cmv)
    return sc_gt