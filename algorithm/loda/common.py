import pyximport; pyximport.install()
from base import *
import numpy as np
import logging
logger = logging.getLogger(__name__)
NA = np.nan #-9999.0

def nrow(x):
    if len(x.shape) == 2:
        return x.shape[0]
    return None
def ncol(x):
    if len(x.shape) == 2:
        return x.shape[1]
    return None

def rbind(m1, m2):
    if m1 is None:
        return np.copy(m2)
    return np.append(m1, m2, axis=0)

def cbind(m1, m2):
    if len(m1.shape) == 1 and len(m2.shape) == 1:
        if len(m1) == len(m2):
            mat = np.empty(shape=(len(m1), 2))
            mat[:, 0] = m1
            mat[:, 1] = m2
            return mat
        else:
            raise ValueError("length of arrays differ: (%d, %d)" % (len(m1), len(m2)))
    return np.append(m1, m2, axis=1)
def matrix(d, nrow=None, ncol=None, byrow=False):
    """Returns the data as a 2-D matrix

    A copy of the same matrix will be returned if input data dimensions are
    same as output data dimensions. Else, a new matrix will be created
    and returned.

    Example:
        d = np.reshape(range(12), (6, 2))
        matrix(d[0:2, :], nrow=2, byrow=True)

    Args:
        d:
        nrow:
        ncol:
        byrow:

    Returns: np.ndarray
    """
    if byrow:
        # fill by row...in python 'C' fills by the last axis
        # therefore, data gets populated one-row at a time
        order = 'C'
    else:
        # fill by column...in python 'F' fills by the first axis
        # therefore, data gets populated one-column at a time
        order = 'F'
    if len(d.shape) == 2:
        d_rows, d_cols = d.shape
    elif len(d.shape) == 1:
        d_rows, d_cols = (1, d.shape[0])
    else:
        raise ValueError("Dimensions more than 2 are not supported")
    if nrow is not None and ncol is None:
        ncol = int(d_rows * d_cols / float(nrow))
    elif ncol is not None and nrow is None:
        nrow = int(d_rows * d_cols / float(ncol))
    if len(d.shape) == 2 and d_rows == nrow and d_cols == ncol:
        return d.copy()
    if not d_rows * d_cols == nrow * ncol:
        raise ValueError("input dimensions (%d, %d) not compatible with output dimensions (%d, %d)" %
                         (d_rows, d_cols, nrow, ncol))
    if isinstance(d, csr_matrix):
        return d.reshape((nrow, ncol), order=order)
    else:
        return np.reshape(d, (nrow, ncol), order=order)

def order(x, decreasing=False):
    if decreasing:
        return np.argsort(-x)
    else:
        return np.argsort(x)

def sample(x, n):
    shuffle = np.array(x)
    np.random.shuffle(shuffle)
    return shuffle[0:n]

# class HistogramR(object):
#     def __init__(self, counts, density, breaks):
#         self.counts = counts
#         self.density = density
#         self.breaks = breaks

def histogram_r(x, g1=1., g2=1., g3=-1., verbose=False):
    """Construct histograms that mimic behavior of R histogram package

    The type of histogram is 'regular', and right-open
    Note: the number of breaks is being computed as in:
    L. Birge, Y. Rozenholc, How many bins should be put in a regular histogram? 2006
    """

    # compute the number of bins based on the formula used by R package
    n = len(x)

    #print np.argwhere(np.isnan(x))

    nbinsmax = int(g1 * (n ** g2) * (np.log(n) ** g3))
    if verbose:
        logger.debug("max bins: %d" % (nbinsmax,))
    # density, breaks = np.histogram(x, bins=nbinsmax, density=True)

    # the below implements Birge technique that recomputes the bin sizes...
    y = np.sort(x)
    likelihood = np.zeros(nbinsmax, dtype=float)
    pen = np.arange(1, nbinsmax + 1, dtype=float) + ((np.log(np.arange(1, nbinsmax + 1, dtype=float))) ** 2.5)
    for d in range(1, nbinsmax + 1):
        # counts, breaks = np.histogram(x, bins=(y[0] + (np.arange(0, d+1, dtype=float)/d) * (y[n-1]-y[0])),
        #                              density=False)
        counts, breaks = np.histogram(x, bins=d, density=False)
        #np.his
        density = counts / (n * (breaks[1] - breaks[0]))
        like = np.zeros(d, dtype=float)
        like2 = np.zeros(d, dtype=float)
        tmp = np.where(counts > 0)[0]
        if len(tmp) > 0:
            like2[tmp] = np.log(density[tmp])
        like[np.isfinite(like2)] = like2[np.isfinite(like2)]
        likelihood[d - 1] = np.sum(counts * like)
        if np.min(counts) < 0:
            likelihood[d - 1] = -np.Inf
    penlike = likelihood - pen
    optd = np.argmax(penlike)
    if verbose:
        logger.debug("optimal num bins: %d" % (optd + 1,))

    counts, breaks = np.histogram(x, bins=optd + 1, density=False)
    density = counts / (n * (breaks[1] - breaks[0]))

    hist = HistogramR(counts=np.array(counts, float), density=np.array(density, float),
                      breaks=np.array(breaks, dtype=float))
    return hist

# def get_bin_for_equal_hist(breaks, x):
#     if np.isnan(x):
#         return 0
#     if x < breaks[0]:
#         return 0
#     if x > breaks[len(breaks)-1]:
#         return len(breaks)-1
#     i = np.trunc((x - breaks[0]) / (breaks[1] - breaks[0]))  # get integral value
#     return int(i)


def pdf_hist_equal_bins(x, h, minpdf=1e-8):
    # here we are assuming a regular histogram where
    # h.breaks[1] - h.breaks[0] would return the width of the bin
    p = (x - h.breaks[0]) / (h.breaks[1] - h.breaks[0])
    ndensity = len(h.density)
    p = np.array([min(int(np.trunc(v)), ndensity-1) for v in p])
    d = h.density[p]
    # quick hack to make sure d is never 0
    d = np.array([max(v, minpdf) for v in d])
    return d

# def pdf_hist(x, h, minpdf=1e-8):
#     if type(x) is np.ndarray:
#         n = len(x)
#         pd = np.zeros(n)
#         for j in range(n):
#             # use simple index lookup in case the histograms are equal width
#             # this returns the lower index
#             i = get_bin_for_equal_hist(h.breaks, x[j])
#             if i >= len(h.density):
#                 i = len(h.density)-1  # maybe something else should be done here
#             # i might be 0 if the value does not lie within the
#             # histogram range. We will assume for now that it cannot happen.
#
#             # More accurately, we should also multiply by diff(h$breaks)[i];
#             # however, all breaks are equal in length in this algorithm,
#             # hence, ignoring that for now.
#             # also, hack to make sure that density is not zero
#             pd[j] = max(h.density[i], minpdf)
#     else:
#         i = get_bin_for_equal_hist(h.breaks, x)
#         if i >= len(h.density):
#             i = len(h.density) - 1
#         pd = max(h.density[i], minpdf)
#
#     return pd

