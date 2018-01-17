from sklearn import mixture
import numpy as np












if __name__ == '__main__':
    import pandas as pd
    from util.common import metric
    df = pd.read_csv('/home/tadeze/projects/missingvalue/datasets/anomaly/yeast/fullsamples/yeast_1.csv')
    train_data = df.ix[:, 1:].as_matrix().astype(np.float64)
    # train_lbl = df.ix[:,0] #
    train_lbl = map(int, df.ix[:, 0] == "anomaly")
    gmm = mixture.GaussianMixture(n_components=3)
    gmm.fit(train_data)
    score = -gmm.score_samples(train_data)
    print gmm.get_params(False)
    print len(score)
    print metric(train_lbl, score)
    gmx = mixture.GaussianMixture()
    print gmm.score(train_data[0,:].reshape([1,train_data.shape[1]])), score[0]