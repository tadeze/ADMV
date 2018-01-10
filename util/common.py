import numpy as np
import sklearn.metrics as mt


## MISSING VALUE FLAGES
MISSING_VALUE = -9999.0
missing_value_per = 0.1
NA = np.nan

def rbind(r1, r2):
    if r1 is None:
        return np.copy(r2)
    return np.append(r1, r2, axis=0)


def cbind(c1, c2):
    if len(c1.shape) == 1 and len(c2.shape) == 1:
        mat = np.empty(shape=(len(c1), 2))
        m[:, 0] = c1
        m[:, 1] = c2
        return mat
    else:
        raise ValueError("length of different")
    return np.append(c1, c1, axis=1)


def metric(label, score):
    auc = mt.roc_auc_score(label, score)
    ap = mt.average_precision_score(label, score)
    return [auc, ap]