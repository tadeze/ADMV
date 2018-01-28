import numpy as np
import sklearn.metrics as mt
#tempresult = "/nfs/guille/bugid/adams/ifTadesse/kddexperiment/missingdata/tempresult"
tempresult = "~/projects/research/kdd2018/tempresult"

## MISSING VALUE FLAGES
MISSING_VALUE = -9999.0
missing_value_per = 0.1
#NA = np.nan
NA = np.nan
def rbind(r1, r2):
    if r1 is None:
        return np.copy(r2)
    return np.append(r1, r2, axis=0)

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
def metric(label, score):
    auc = mt.roc_auc_score(label, score)
    ap = mt.average_precision_score(label, score)
    return [auc, ap]
def get_miss_features(row):
    """
    :param row: np.ndarray of 1Xd vector.
    :return:
    """
    if np.isnan(NA):
        miss_column = np.where(np.isnan(row))[0]
    else:
        miss_column = np.where(row == NA)[0]
    return miss_column
def available_models(models_featues, miss_column):
    """
    Return trees without the missing column
    :param miss_column:
    :return:
    """
    if len(miss_column)==0:
        return range(0,models_featues.shape[0])
    with_miss_column = models_featues[:,miss_column]
    available_models = np.where(~with_miss_column.any(axis=1))[0]
    return available_models
if __name__ == '__main__':
    print metric([1,0,1,0,0],[0.3,0.5,0.4,0.2,0.9])