
import numpy as np
from missvalueinjector import MissingValueInjector
from mainexperiment import algo_miss_features, random_miss_prop
import pandas as pd
def test():
    df = pd.read_csv("yeast_1.csv")
    train_data = df.ix[:,1:].as_matrix()
    train_lbl = map(int, df.ix[:, 0] == "anomaly")
    miss_colmn = range(0,5)

    # result = algo_miss_features(
    #     train_data, train_lbl, "yeast.csv", miss_colmn,'IFOR')
    result = random_miss_prop(
        train_data, train_lbl, 'BIFOR',miss_colmn,)
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