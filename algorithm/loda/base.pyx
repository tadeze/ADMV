cimport numpy
#from libc.math import isnan
cimport libc.math as cmath
import numpy as np

cdef class HistogramR(object):
     cdef public numpy.ndarray counts
     cdef public numpy.ndarray breaks
     cdef public numpy.ndarray density
     def __cinit__(self, counts, density, breaks):
        self.density = density
        self.counts = counts
        self.breaks = breaks
#counts=np.array(counts, float), density=np.array(density, float),
#                      breaks=np.array(breaks, dtype=float)
#
# class HistogramR(object):
#     def __init__(self, counts, density, breaks):
#         self.counts = counts
#         self.density = density
#         self.breaks = breaks

cdef int get_bin_for_equal_hist(double* breaks, double x, int m) nogil:
    if cmath.isnan(x):
        return 0
    if x<breaks[0]:
        return 0
    if x > breaks[m-1]:
        return m-1
    #with gil:
    cdef int i = <int>cmath.floor((x - breaks[0]) / (breaks[1] - breaks[0]))  # get integral value
    return i

# cdef pdf_hist_equal_bins(double x, Histogram* h, double minpdf=1e-8):
#     # here we are assuming a regular histogram where
#     # h.breaks[1] - h.breaks[0] would return the width of the bin
#     p = (x - h.breaks[0]) / (h.breaks[1] - h.breaks[0])
#     ndensity = len(h.density)
#     p = np.array([min(int(np.trunc(v)), ndensity-1) for v in p])
#     d = h.density[p]
#     # quick hack to make sure d is never 0
#     d = np.array([max(v, minpdf) for v in d])
#     return d
#

cpdef pdf_hist_w(numpy.ndarray x, numpy.ndarray breaks, numpy.ndarray density, double minpdf ):

    cdef double* hist_break = <double*> breaks.data
    cdef int hist_size = breaks.shape[0]
    cdef int hist_den_size = density.shape[0]
    cdef int n = <int>x.size
    cdef numpy.ndarray pd = np.zeros(n)
    cdef int  i
    if n>1:

        for j in range(n):
            # use simple index lookup in case the histograms are equal width
            # this returns the lower index
            i = get_bin_for_equal_hist(hist_break, x[j], hist_size)
            if i >= hist_den_size:
                i = hist_den_size-1  # maybe something else should be done here
            pd[j] = max(density[i], minpdf)
    else:
         i = get_bin_for_equal_hist(hist_break, x, hist_size)
         if i >= hist_den_size:
             i = hist_den_size - 1
         pd[0] = max(density[i], minpdf)
    return pd


cpdef pdf_hist(x, HistogramR h, double minpdf=1e-8):

    #pdf_hist(x, h.breaks, h.density, minpdf)

    if type(x) is np.ndarray:
        pd = pdf_hist_w(x, h.breaks, h.density, minpdf)

    else:
        y = np.array(x)
        pd = pdf_hist_w(y, h.breaks, h.density, minpdf)
    return pd

cpdef numpy.ndarray cy_get_all_hist_pdfs_miss(numpy.ndarray a, numpy.ndarray w, list hists):
    cdef int  a_c = a.shape[0], hist_len = len(hists)
    # x = a.dot(w)
    cpdef numpy.ndarray hpdfs = np.zeros(shape=(len(a), len(hists)), dtype=float)
    cdef numpy.ndarray x
    cdef numpy.ndarray miss_column
    for i in xrange(0, a.shape[0]):
        miss_column = np.where(np.isnan((a[i,:])))[0]
        #miss_column = np.where(a[i, :] == NA)[0]
        # search w with this projections
        if miss_column.size > 0:
            w_miss = w[miss_column, :]
            idx_miss = np.where(~w_miss.any(axis=0))[0] #.tolist()
            temp = a[i, :].copy()  ## small hack to replace nan with any number because nan*0 doesn't give real number.
            temp[np.isnan(temp)] = -9999999999999.0
            x = temp.dot(w[:, idx_miss])

            for ix, ihist in enumerate(idx_miss):
                hpdfs[i, ihist] = pdf_hist(x[ix], hists[ihist])
        else:
            x = a[i, :].dot(w)
            for ihist in range(len(hists)):
                hpdfs[i, ihist] = pdf_hist(x[ihist], hists[ihist])
    return hpdfs