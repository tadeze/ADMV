
import numpy as np
from missvalueinjector import MissingValueInjector
from mainexperiment import algo_miss_features, random_miss_prop, algo_miss_featuresX
import algorithm.pyad as pft
import pandas as pd
from algorithm.lof import BaggedLOF
from util.common import metric
import time
np.random.seed(100)
def test():
    #file_name = "/nfs/guille/bugid/adams/ifTadesse/missingdata/experiments/anomaly/shuttle_1v23567/fullsamples/shuttle_1v23567_1.csv"
    #file_name = "/home/tadeze/projects/missingvalue/datasets/anomaly/yeast/fullsamples/yeast_1.csv"
    file_name ="yeast_1.csv"
    df = pd.read_csv(file_name)
    train_data = df.ix[:,1:].as_matrix().astype(np.float64)
    #train_lbl = df.ix[:,0] #
    train_lbl =  map(int, df.ix[:, 0] == "anomaly")
    miss_colmn = range(0,5)
    #result = algo_miss_features(
     #    train_data, train_lbl, miss_colmn,'LODA')
    # start = time.time()
    # result = algo_miss_features(
    #      train_data, train_lbl, miss_colmn,'egmm',file_name)
    # print result
    #
    # print time.time() - start
    start = time.time()
    result = algo_miss_featuresX(
        train_data, train_lbl, miss_colmn, 'ifor', file_name)
    print time.time() - start
    print result

def test_loda():
    pass
def check_metric(data_path):

    df = pd.read_csv(data_path)
    train_data = df.ix[:, 5:].as_matrix().astype(np.float64)
    # train_lbl = df.ix[:,0] #
    train_lbl = map(int, df.ix[:, 4] == "anomaly")
    ff = pft.IsolationForest(ntree=100)
    ff= BaggedLOF()
    ff.train(train_data)
    print metric(train_lbl, ff.score(train_data))
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
    #dd = 'concrete'
    #check_metric('/nfs/guille/bugid/adams/meta_analysis/mothersets/regression/'+dd+'/'+dd+'.preproc.csv')
    #test_cell_injector()
