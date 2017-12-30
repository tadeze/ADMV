import pyad as pft

def train_forest(train_x, check_mv=False, tree_size=1):
    ff = pft.IsolationForest(train_x, ntree=100 * tree_size, nsample=512)
    return ff

def score(test_x, cmv=False):
    sc_gt = ff.score(test_x, cmv)
    return sc_gt