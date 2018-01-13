
import numpy as np
from missvalueinjector import MissingValueInjector
from mainexperiment import algo_miss_features, random_miss_prop
import pandas as pd
def test():
    file_name = "/nfs/guille/bugid/adams/ifTadesse/missingdata/experiments/anomaly/shuttle_1v23567/fullsamples/shuttle_1v23567_1.csv"
    df = pd.read_csv(file_name)
    train_data = df.ix[:,1:].as_matrix().astype(np.float64)
    train_lbl = map(int, df.ix[:, 0] == "anomaly")
    miss_colmn = range(0,5)

    result = algo_miss_features(
         train_data, train_lbl, miss_colmn,'LODA')
    #result = random_miss_prop(
    #     train_data, train_lbl, miss_colmn,'IFOR')
    print result
def test_cell_injector():
    w = np.random.randn(10,5)
    ad_in = MissingValueInjector()
    ix = ad_in.inject_missing_in_random_cell(w,0.3)
    print w
    print ix


if __name__ == '__main__':
    test()
    #test_cell_injector()