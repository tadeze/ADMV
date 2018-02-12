
import numpy as np
from missvalueinjector import MissingValueInjector
#from mainexperiment import algo_miss_features, random_miss_prop, algo_miss_featuresX
import algorithm.pyad as pft
import pandas as pd
from algorithm.lof import BaggedLOF
from util.common import metric
import time
from algorithm.egmm import Egmm
from splitjobs import single_benchmark, algo_miss_featuresX
np.random.seed(100)
def test():
    #file_name = "/nfs/guille/bugid/adams/ifTadesse/missingdata/experiments/anomaly/shuttle_1v23567/fullsamples/shuttle_1v23567_1.csv"
<<<<<<< HEAD
    file_name = "yeast_1.csv"

=======
    #file_name = "/home/tadeze/projects/missingvalue/datasets/anomaly/shuttle_1v23567/fullsamples/shuttle_1v23567_1.csv"
    file_name ="../group2/wave_benchmark_1562.csv"
    #file_name ="yeast_1.csv"
>>>>>>> serialbase
    df = pd.read_csv(file_name)
    train_data = df.ix[:,6:].as_matrix().astype(np.float64)
    #train_lbl = df.ix[:,0] #
<<<<<<< HEAD
    train_lbl =  map(int, df.ix[:, 0] == "anomaly")
    miss_colmn = range(0,5)

    result = algo_miss_features(
         train_data, train_lbl, miss_colmn,'ifor')
    #result = random_miss_prop(
     #    train_data, train_lbl, miss_colmn,'ifor')
    print result
=======
    train_lbl =  map(int, df.ix[:, 5] == "anomaly")
    miss_colmn = range(0,6)
    #result = algo_miss_features(
     #    train_data, train_lbl, miss_colmn,'LODA')
    start = time.time()
    # result = algo_miss_features(
    #      train_data, train_lbl, miss_colmn,'egmm',file_name)
    # print result
    #
    # print time.time() - start
    #algo_list =
    # start = time.time()
    # #for algo in ['ifor','bifor','loda']:
    #     #print algo
    # #single_benchmark(train_x, label, miss_column, file_name, label_field, algorithm_list=ALGORITHMS, task_id=1):
    # for algo in ["IFOR","BIFOR","LODA"]:
    #     result = single_benchmark(train_x=train_data,label=train_lbl, miss_column=miss_colmn,file_name=file_name,
    #                           label_field=1,algorithm_list=[algo])
    #
    # print time.time() - start
    # start = time.time()
    # #print result
    # #print result

    # result = algo_miss_featuresX(
    #      train_data, train_lbl, miss_colmn, 'egmm', file_name)
    result = single_benchmark(train_x=train_data, label=train_lbl, miss_column=miss_colmn, file_name=file_name,
                              label_field=0,algorithm_list=["egmm"],task_id=14)
    #TODO: Debug the egmm code.
    print time.time() - start
    print pd.DataFrame(result).head(5)

    #print result


def test_eggm():
    file_name ="mixturesynthetic/synthetic_mixture_d_8_delta_1_1_rho_1.6iter_5.csv"
    gmm = Egmm()
    gmm.train(file_name,7,"aba.mdl","trainout.csv",skip_cols=1)

    score = gmm.score_file(file_name,7,"aba.mdl", "score_out.csv", skip_cols=1)
    df = pd.read_csv(file_name)
    #print df.head(5)
    ss = pd.read_csv("tempresult/trainout.csv")
    ss["logscore"] = -1*np.log(-1*ss["score"])
    ss = ss.sort_values("id")
    ss.to_csv("tempresult/trainout-order.csv")
    train_lbl = map(int, df.ix[:, 5] == "anomaly")
    print train_lbl
    print metric(train_lbl, score)
    print metric(train_lbl, ss["logscore"].as_matrix())


>>>>>>> serialbase
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
<<<<<<< HEAD
    test()
    #dd = 'concrete'
    #check_metric('/nfs/guille/bugid/adams/meta_analysis/mothersets/regression/'+dd+'/'+dd+'.preproc.csv')
    #test_cell_injector()
=======
    #test()
    test_eggm()
    #dd = 'concrete'
    #check_metric('/nfs/guille/bugid/adams/meta_analysis/mothersets/regression/'+dd+'/'+dd+'.preproc.csv')
    #test_cell_injector()
>>>>>>> serialbase
