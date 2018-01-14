
import numpy as np
from missvalueinjector import MissingValueInjector
from mainexperiment import algo_miss_features, random_miss_prop
import algorithm.pyad as pft
import pandas as pd
def test():
    file_name = "yeast_1.csv"
    #file_name = "/nfs/guille/bugid/adams/ifTadesse/missingdata/experiments/synthetic100.csv"
    df = pd.read_csv(file_name)
    train_data = df.ix[:,1:].as_matrix().astype(np.float64)
    #train_lbl = df.ix[:,0] #
    train_lbl =  map(int, df.ix[:, 0] == "anomaly")
    miss_colmn = range(0,5)

    #result = algo_miss_features(
     #    train_data, train_lbl, miss_colmn,'iFor')
    result = random_miss_prop(
         train_data, train_lbl, miss_colmn,'bifor')
    print result
def test_loda():
    pass
def test_cell_injector():
    w = np.random.randn(10,5)
    test = w.copy()
    ad_in = MissingValueInjector()
    ix = ad_in.inject_missing_in_random_cell(w,0.3)
   # print w
    #print ix
    ff = pft.IsolationForest(ntree=10)
    ff.train(test)
    print ff.score(test)
    print ff.average_depth()[0:3]

    print ff.score(w)
    print ff.average_depth()[0:3]

    print ff.score(w, cmv=True)
    print ff.average_depth()[0:3]
    print pft.__file__

    #print pft.__file__
if __name__ == '__main__':
    test()
    #test_cell_injector()