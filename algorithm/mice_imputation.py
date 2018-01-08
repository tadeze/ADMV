import fancyimpute as fi



def impute_value(df):
    mice_impute = fi.MICE().complete(df)
    return mice_impute

import numpy as np
if __name__ == '__main__':
    w = np.random.randn(10,2)
    w_n = w.copy()
    w_n[3,1] = np.nan
    w_c = fi.MICE().complete(w_n)
    print w_c, w