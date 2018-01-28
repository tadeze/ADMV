import pyximport; pyximport.install()
import base as pyx

from common import *
#NA = np.nan
class ProjectionVectorsHistograms(object):
    def __init__(self, w=None, hists=None):
        """

        Args:
            w: numpy.ndarray(dtype=int)
            hists: list of HistogramR
        """
        self.w = w
        self.hists = hists

class LodaModel(object):
    def __init__(self, k=0, pvh=None, sigs=None):
        """
        Args:
            k: int
            pvh: ProjectionVectorsHistograms
            sigs: numpy.array(dtype=float)
        """
        self.k = k
        self.pvh = pvh
        self.sigs = sigs

class LodaResult(object):
    def __init__(self, anomranks=None, nll=None, pvh=None):
        """

        Args:
            anomranks: numpy.array(dtype=int)
            nll: numpy.array(dtype=float)
            pvh: LodaModel
        """
        self.anomranks = anomranks
        self.nll = nll
        self.pvh = pvh

class Loda(object):
    def __init__(self, sparsity=np.nan, mink=1, maxk=0,
    keep=None, exclude=None, original_dims=False):
        self.sparsity = sparsity
        self.mink = mink
        self.maxk = maxk
        self.keep = keep
        self.exclude = exclude
        self.original_dims = original_dims

    def get_random_proj(self, nproj, d, sp, keep=None, exclude=None):
        nzeros = int(np.floor(d * sp))
        idxs = np.arange(d)  # set of dims that will be sampled to be set to zero
        marked = []
        if keep is not None:
            marked.extend(keep)
        if exclude is not None:
            # since 'exclude' contains the dims that are
            # predetermined to be zero, adjust the number
            # of zero dims that need to be further determined
            # by sampling
            nzeros -= len(exclude)
            marked.extend(exclude)
        if len(marked) > 0:
            # remove from the known set -- the dims that have been
            # marked for keeping or excluding. There is no uncertainty in
            # the selection/rejection of marked dims.
            idxs = np.delete(idxs, marked)
        w = np.zeros(shape=(d, nproj), dtype=float)
        for i in range(nproj):
            w[:, i] = np.random.randn(d)
            if nzeros > 0:
                z = sample(idxs, min(nzeros, len(idxs)))
                # shuffle = np.array(idxs)
                # np.random.shuffle(shuffle)
                # z = shuffle[0:min(nzeros, len(idxs))]
                if exclude is not None:
                    z.extend(exclude)
                w[z, i] = 0
            w[:, i] = w[:, i] / np.sqrt(sum(w[:, i] * w[:, i]))

        return w

    def build_proj_hist(self, a, w):
        d = ncol(w)  # number of columns
        x = a.dot(w)
        hists = []
        for j in range(d):
            hists_j = histogram_r(x[:, j])
            hists.append(hists_j)
        return hists

    def get_neg_ll(self, a, w, hist, inf_replace=np.nan):
        x = a.dot(w)
        pdfs = np.zeros(shape=(len(x), 1), dtype=float)
        pdfs[:, 0] = pdf_hist_equal_bins(x, hist)
        pdfs[:, 0] = np.log(pdfs)[:, 0]
        if inf_replace is not np.nan:
            pdfs[:, 0] = [max(v, inf_replace) for v in pdfs[:, 0]]
        return -pdfs  # neg. log-lik of pdf

    # get all pdfs from individual histograms.
    def get_all_hist_pdfs(self, a, w, hists):
        x = a.dot(w)
        hpdfs = np.zeros(shape=(len(x), len(hists)), dtype=float)
        for i in range(len(hists)):
            hpdfs[:, i] = pdf_hist(x[:, i], hists[i])
        return hpdfs


    # Compute negative log-likelihood using random projections and histograms
    def get_neg_ll_all_hist(self, a, w, hists, inf_replace=np.nan, check_miss=False):
        if check_miss:
            pds = self.get_all_hist_pdfs_miss(a, w, hists)
        else:
            pds = self.get_all_hist_pdfs(a, w, hists)
        ll = self.get_score(pds, inf_replace)
        # pds = get_all_hist_pdfs(a,w,hists)
        ## Average only using the
        # pds = np.log(pds)
        # if inf_replace is not np.nan:
        #     vfunc = np.vectorize(max)
        #     pds = vfunc(pds, 1.0 * inf_replace) # [max(v, inf_replace) for v in pds[:, i]]
        # ll = -np.mean(pds, axis=1)  # neg. log-lik
        return ll

    def get_score(self, pds, inf_replace):

        #  ll_score = np.zeros_like(pds)
        # pds[0,:] = np.zeros(pds.shape[1])
        pds_log = pds.copy()
        pds_log[pds_log == 0] = 1
        pds_log = np.log(pds_log)
        # print pds
        if inf_replace is not np.nan:
            vfunc = np.vectorize(max)
            pds = vfunc(pds, 1.0 * inf_replace)  # [max(v, inf_replace) for v in pds[:, i]]
        non_zero_proj = np.count_nonzero(pds, axis=1)
        # non_zero_proj[0] =0
        non_zero_proj[non_zero_proj == 0] = 1
        # print non_zero_proj
        ll = -np.sum(pds_log, axis=1) / non_zero_proj  # -np.mean(pds, axis=1)  # neg. log-lik
        return ll

    def get_miss_features(self, test_df):
        if np.isnan(NA):
            miss_column = np.where(np.isnan(test_df))[0]
        else:
            miss_column = np.where(test_df == NA)[0]
        return miss_column
    def get_all_hist_pdfs_miss(self, a, w, hists):

        # x = a.dot(w)
        # hpdfs = np.zeros(shape=(len(a), len(hists)), dtype=float)
        #
        # for i in range(0, a.shape[0]):
        #     miss_column = self.get_miss_features(a[i,:])
        #     #miss_column = np.where(a[i, :] == NA)[0]
        #     # search w with this projections
        #     if len(miss_column) > 0:
        #         w_miss = w[miss_column, :]
        #         idx_miss = np.where(~w_miss.any(axis=0))[0].tolist()
        #         temp = a[i, :].copy()  ## small hack to replace nan with any number because nan*0 doesn't give real number.
        #         temp[np.isnan(temp)] = -9999999999999.0
        #         x = temp.dot(w[:, idx_miss])
        #
        #         for ix, ihist in enumerate(idx_miss):
        #             hpdfs[i, ihist] = pdf_hist(x[ix], hists[ihist])
        #     else:
        #         x = a[i, :].dot(w)
        #         for ihist in range(len(hists)):
        #             hpdfs[i, ihist] = pdf_hist(x[ihist], hists[ihist])
        # return hpdfs
        return cy_get_all_hist_pdfs_miss(a, w, hists)

    # Determine k - no. of dimensions
    # sp=1 - 1 / np.sqrt(ncol(a)),
    def get_best_proj(self, a, mink=1, maxk=10, sp=0.0, keep=None, exclude=None):
        """

        :type a: numpy.ndarray
        :type mink: int
        :type maxk: int
        :type sp: float
        """
        t = 0.01
        n = nrow(a)
        d = ncol(a)

        # if (debug) print(paste("get_best_proj",maxk,sp))
        # logger.debug("get_best_proj: sparsity: %f" % (sp,))

        w = np.zeros(shape=(d, maxk + 1), dtype=float)
        hists = []
        fx_k = np.zeros(shape=(n, 1), dtype=float)
        fx_k1 = np.zeros(shape=(n, 1), dtype=float)

        w_ = self.get_random_proj(nproj=1, d=d, sp=sp, keep=keep, exclude=exclude)
        w[:, 0] = w_[:, 0]
        hists.append(self.build_proj_hist(a, w_)[0])
        fx_k[:, 0] = self.get_neg_ll(a, w_, hists[0])[:, 0]

        sigs = np.ones(maxk) * np.Inf
        k = 0
        # logger.debug("mink: %d, maxk: %d" % (mink, maxk))
        while k <= mink or k < maxk:
            w_ = self.get_random_proj(nproj=1, d=d, sp=sp, keep=keep, exclude=exclude)
            w[:, k + 1] = w_[:, 0]
            hists.append(self.build_proj_hist(a, w_)[0])

            ll = self.get_neg_ll(a, w[:, k + 1], hists[k + 1])

            fx_k1[:, 0] = fx_k[:, 0] + ll[:, 0]

            diff_ll = abs(fx_k1 / (k + 2.0) - fx_k / (k + 1.0))
            # logger.debug(diff_ll)
            diff_ll = diff_ll[np.isfinite(diff_ll)]
            if len(diff_ll) > 0:
                sigs[k] = np.mean(diff_ll)
            else:
                raise (ValueError("Log-likelihood was invalid for all instances"))
            tt = sigs[k] / sigs[0]
            # print (c(tt, sigs[k], sigs[1]))
            # print(which(is.na(diff_ll)))
            # print(diff_ll)
            if tt < t and k >= mink:
                break

            fx_k[:, 0] = fx_k1[:, 0]

            # if (debug) print(paste("k =",k,"; length(sigs)",length(sigs),"; sigs_k=",tt))

            k += 1

        bestk = np.where(sigs == np.min(sigs))[0][0]  # np.where returns tuple of arrays
        # print "bestk: %d" % (bestk,)
        return LodaModel(bestk, ProjectionVectorsHistograms(matrix(w[:, 0:bestk], nrow=nrow(w)),
                                                            hists[0:bestk]),
                         sigs)

    def get_original_proj(self, a, maxk=10, sp=0, keep=None, exclude=None):
        d = ncol(a)
        w = np.zeros(shape=(d, d - (0 if exclude is None else len(exclude))), dtype=float)
        hists = []
        k = 0
        for l in range(d):
            if exclude is not None and len(np.where(exclude == l)[0]) > 0:
                continue
            w_ = np.zeros(shape=(d, 1), dtype=float)
            w_[l, 0] = 1  # important: the 'l'-th (not 'k'-th) dim is 1
            w[:, k] = w_[:, 0]
            hists.append(self.build_proj_hist(a, w_)[0])
            k += 1

        return LodaModel(k=k, pvh=ProjectionVectorsHistograms(w=w, hists=hists), sigs=None)


    def train(self, train_x):
        l, d = train_x.shape
        self.maxk = max(self.mink, 3*d if self.maxk ==0 else self.maxk)
        if self.sparsity is None or self.sparsity is np.nan:
            sp = 0 if d == 1 else 1 - 1/np.sqrt(d)
        else:
            sp = self.sparsity
        logger.debug("loda: sparsity: %f" % sp)
        if self.original_dims:
            pvh = self.get_original_proj(train_x,self.maxk,sp,self.keep,self.exclude)
        else:
            pvh = self.get_best_proj(train_x,self.mink,self.maxk,sp,self.keep,self.exclude)
        self.pvh = pvh
    def score(self, test_x, check_miss=False):
        nll = self.get_neg_ll_all_hist(test_x, self.pvh.pvh.w, self.pvh.pvh.hists, inf_replace = np.nan, check_miss = check_miss)
        anom_ranks = np.arange(nrow(test_x))
        anom_ranks = anom_ranks[order(-nll)]
        return nll #LodaResult(anomranks=anom_ranks, nll=nll, pvh=self.pvh)


