from lodaor import *
import numpy as np
from loda import Loda
class lodax:

    def __init__(self):
        self.pvh = None
    def train(self, train_x, sparsity=np.nan, mink=1, maxk=0, keep=None, exclude=None, original_dims=False):
        l, d = train_x.shape
        maxk = 3 * d if maxk == 0 else maxk
        maxk = max(mink, maxk)

        if sparsity is None or sparsity is np.nan:
            sp = 0 if d == 1 else 1 - 1 / np.sqrt(d)
        else:
            sp = sparsity

        if original_dims:
            self. pvh = get_original_proj(train_x, maxk=maxk, sp=sp, keep=keep, exclude=exclude)
        else:
            self.pvh = get_best_proj(train_x, mink=mink, maxk=maxk, sp=sp, keep=keep, exclude=exclude)


    def score(self, test_x, check_miss=False):
         nll = get_neg_ll_all_hist(test_x, self.pvh.pvh.w, self.pvh.pvh.hists,
                                   inf_replace=np.nan, check_miss=check_miss)
         return nll


if __name__ == '__main__':
    import pandas as pd
    import util.common as cm
    from missvalueinjector import MissingValueInjector
    mvi = MissingValueInjector()

    df = pd.read_csv('../../yeast_1.csv')
    train_data = df.ix[:, 1:].as_matrix().astype(np.float64)
    # train_lbl = df.ix[:,0] #
    train_lbl = map(int, df.ix[:, 0] == "anomaly")
    print df.head(5)
    ll = Loda(maxk=100)
    ll.train(train_data)

    ldx = lodax()
    ldx.train(train_data,maxk=100)
    score = ll.score(train_data, True)
    print cm.metric(train_lbl, score)
    mvi.inject_missing_value(train_data, 2, 0.5,range(1,train_data.shape[1]))
    #train_data[0,2] = np.nan
    #train_data[11:100,[3,4]] = np.nan
    score = ll.score(train_data, True)
    print cm.metric(train_lbl, score)

    score = ll.score(train_data, False)
    print cm.metric(train_lbl, score)

    train_data[np.isnan(train_data)] = -9999.0 #MISSING_VALUE
    #train_data[0, 2] = -999.0
    #train_data[11:100, [3, 4]] = -999.0
    sc = ldx.score(train_data, True)
    print cm.metric(train_lbl, sc)


    # def loda(train, test=None, sparsity=np.nan, mink=1, maxk=0, keep=None, exclude=None, original_dims=False,
    #          check_miss=False):
    #     if test is None:
    #         test = train
    #     #assert ncol(train) == ncol(test)
    #     l, d = train.shape
    #     maxk = 3 * d if maxk == 0 else maxk
    #     maxk = max(mink, maxk)
    #
    #     if sparsity is None or sparsity is np.nan:
    #         sp = 0 if d == 1 else 1 - 1 / np.sqrt(d)
    #     else:
    #         sp = sparsity
    #
    #
    #     if original_dims:
    #         pvh = get_original_proj(train, maxk=maxk, sp=sp, keep=keep, exclude=exclude)
    #     else:
    #         pvh = get_best_proj(train, mink=mink, maxk=maxk, sp=sp, keep=keep, exclude=exclude)
    #
    #     nll = get_neg_ll_all_hist(test, pvh.pvh.w, pvh.pvh.hists, inf_replace=np.nan, check_miss=check_miss)
    #
    #     anomranks = np.arange(l)
    #     anomranks = anomranks[order(-nll)]
    #
    #     return LodaResult(anomranks=anomranks, nll=nll, pvh=pvh)
    #
    # ld = loda.Loda()
    # ld.train(train_x=train_data)
    # # print pvh.pvh
    # score = ld.score(test_data)