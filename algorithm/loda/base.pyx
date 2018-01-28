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

cdef double cmax(double x, double y) nogil:
    if x>y:
        return x
    else:
        return x

cdef numpy.ndarray pdf_hist_w(numpy.ndarray x, int n, HistogramR h, double minpdf, numpy.ndarray pd):
                              #numpy.ndarray breaks, numpy.ndarray density, double minpdf, numpy.ndarray pd):

    cdef double* hist_break = <double*> h.breaks.data
    cdef double* hist_den = <double*> h.density.data
    cdef int hist_size = h.breaks.shape[0]
    cdef int hist_den_size = h.density.shape[0]
    cdef int i

    cdef double* _x = <double*> x.data


    #cdef numpy.ndarray pd = np.zeros(n)
    cdef double* _pd = <double*> pd.data
    if n>1:
        #with nogil:

        for j in range(n):
            # use simple index lookup in case the histograms are equal width
            # this returns the lower index
            i = get_bin_for_equal_hist(hist_break, x[j], hist_size)
            if i >= hist_den_size:
                i = hist_den_size-1  # maybe something else should be done here
            _pd[j] = cmax(hist_den[i], minpdf)
    else:
         #x = (double*)x

         i = get_bin_for_equal_hist(hist_break, x, hist_size)
         if i >= hist_den_size:
             i = hist_den_size - 1
         _pd[0] = cmax(hist_den[i], minpdf)
    return pd


cpdef pdf_hist(x, HistogramR h, double minpdf=1e-8):

    #pdf_hist(x, h.breaks, h.density, minpdf)

    if type(x) is not numpy.ndarray:
        x = np.array(x) #double:

    cdef int n = x.size
    cdef numpy.ndarray pd = np.zeros(n)

    #with nogil:
    return pdf_hist_w(x, n, h, minpdf, pd)


cdef all_hist_pdfs_miss(numpy.ndarray a, list hists, numpy.ndarray w, int n):
    cdef numpy.ndarray miss_column = np.where(np.isnan((a)))[0]
    #miss_column = np.where(a[i, :] == NA)[0]
        # search w with this projections
    cdef numpy.ndarray hpdfs = np.zeros(shape=(n))
    if miss_column.size > 0:

        w_miss = w[miss_column, :]
        idx_miss = np.where(~ np.any(w_miss, axis=0))[0] #.tolist()
        #print len(idx_miss), " Number of available projects"
        temp = a.copy()  ## small hack to replace nan with any number because nan*0 doesn't give real number.
        temp[np.isnan(temp)] = -9999999999999.0
        #x = temp.dot(w[:, idx_miss])
        #print idx_miss, " --available index"
        if len(idx_miss)<1:
            print a
        x = np.dot(temp, w[:,idx_miss])

        for ix, ihist in enumerate(idx_miss):
            hpdfs[ihist] = pdf_hist(x[ix], hists[ihist])
    else:
        x = np.dot(a,w) #a.dot(w)
        for ihist in range(0,n):
            hpdfs[ihist] = pdf_hist(x[ihist], hists[ihist])
    return hpdfs


cpdef numpy.ndarray cy_get_all_hist_pdfs_miss(numpy.ndarray a, numpy.ndarray w, list hists):
    cdef int  a_c = a.shape[0], hist_len = len(hists)
    # x = a.dot(w)
    cpdef numpy.ndarray hpdfs = np.zeros(shape=(len(a), len(hists)), dtype=float)
    cdef numpy.ndarray x
    cdef numpy.ndarray miss_column



    for i in range(0, a_c):
        #miss_column = np.where(np.isnan((a[i,:])))[0]
        #miss_column = np.where(a[i, :] == NA)[0]
        # search w with this projections
        # if miss_column.size > 0:
        #     w_miss = w[miss_column, :]
        #     idx_miss = np.where(~w_miss.any(axis=0))[0] #.tolist()
        #     temp = a[i, :].copy()  ## small hack to replace nan with any number because nan*0 doesn't give real number.
        #     temp[np.isnan(temp)] = -9999999999999.0
        #     x = temp.dot(w[:, idx_miss])
        #
        #     for ix, ihist in enumerate(idx_miss):
        #         hpdfs[i, ihist] = pdf_hist(x[ix], hists[ihist])
        # else:
        #     x = a[i, :].dot(w)
        #     for ihist in range(len(hists)):
        #         hpdfs[i, ihist] = pdf_hist(x[ihist], hists[ihist])
        hpdfs[i,:]= all_hist_pdfs_miss(a[i,:], hists, w, hist_len)

    return hpdfs