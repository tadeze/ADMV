import numpy as np
import pandas as pd

file_description = {'skin': 3,
                    'magic.gamma': 10, 'particle': 50, 'spambase': 57, 'fault': 27, 'gas': 128,
                    'imgseg': 18, 'landsat': 36, 'letter.rec': 16, 'opt.digits': 61, 'pageb': 10, 'shuttle': 9,
                    'wave': 21, 'yeast': 8, 'comm.and.crime': 101, 'abalone': 7, 'concrete': 8, 'wine': 11, 'yearp': 90,
                    'synthetic': 8
                    }



def entropy(sigma):
    _entropy = 0.5*np.log(2*np.pi*np.exp(1)*np.linalg.det(sigma))
    return _entropy
def total_correlation(mat_data):
    print mat_data.shape
    joint_entropy = entropy(np.cov(mat_data.T))
    print "Joint -=- ",joint_entropy
    marg_entrop = np.sum(entropy(np.reshape(icov,[1,1])) for icov in np.var(mat_data,axis=0))
    print "marginal -- ", marg_entrop
    return marg_entrop - joint_entropy


if __name__ == '__main__':
    w = np.random.randn(3,10)
    dat = pd.read_csv('../group2/concrete_benchmark_1524.csv')
    w = dat[dat['ground.truth']=="nominal"].ix[:,6:13].as_matrix().astype(np.float64)
    #print w.shape
    cvv = np.cov(w.T)
    print cvv
    #print np.var(w,axis=0)
    np.linalg.det(cvv
    varss = np.var(w, axis=0)
    print varss[0], np.sum([entropy(np.reshape(vv,[1,1])) for vv in varss] )#print entropy(cvv))
    #print
    print total_correlation(w)