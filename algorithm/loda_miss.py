from pyloda
from metrics import *

#np.random.seed(100)
def loda_train(train, sparsity=np.nan, mink=1, maxk=0, keep=None, exclude=None,
               original_dims=False):
    l = nrow(train)
    d = ncol(train)

    maxk = 3 * d if maxk == 0 else maxk
    maxk = max(mink, maxk)

    if sparsity is None or sparsity is np.nan:
        sp = 0 if d == 1 else 1 - 1 / np.sqrt(d)
    else:
        sp = sparsity

    logger.debug("loda: sparsity: %f" % (sp,))

    if original_dims:
        pvh = get_original_proj(train, maxk=maxk, sp=sp, keep=keep, exclude=exclude)
    else:
        pvh = get_best_proj(train, mink=mink, maxk=maxk, sp=sp, keep=keep, exclude=exclude)

    #nll = get_neg_ll_all_hist(test, pvh.pvh.w, pvh.pvh.hists, inf_replace=np.nan, check_miss=check_miss)
    return pvh

def loda_score(test,pvh,check_miss=False):
    l = nrow(test)
    d = ncol(test)
    nll = get_neg_ll_all_hist(test, pvh.pvh.w, pvh.pvh.hists, inf_replace=np.nan, check_miss=check_miss)
    anomranks = np.arange(l)
    anomranks = anomranks[order(-nll)]
    return LodaResult(anomranks=anomranks, nll=nll, pvh=pvh)