from sklearn.mixture import GaussianMixture
import numpy as np
from pypr.clustering import gmm
class Box:
    def __init__(self):
        self.h = None
        self.w = None
class GMM(object):
    def __init__(self, replica=30, max_iteration=50, comp=(3,4,5)):
        self.comp = comp
        self.replica = replica
        self.max_iteration = max_iteration
        self.gmm = None
        self.parameters = []
    def train(self, X):
        """
        Train multiple components
        :param X:
        :return:
        """
        for model in range(0, self.replica):
            cmp = np.random.choice(self.comp, 1, replace=True)
            print cmp[0]
            self.parameters[model] = gmm.em_gm(X=X,K=int(cmp[0]), max_iter=self.max_iteration)
    def score(self, X):
        score_list = []
        for param in self.parameters:
            score_list.append(self.__score(X, param))
        return np.mean(np.array(score_list),axis=1)
    def __score(self, X, param):
        if X.ndim>2:
            nll = [gmm.gm_log_likelihood(xi,center_list=param[0],
                                     cov_list=param[1], p_k=param[2]) for xi in X]
            return nll
        else:
            return gmm.gm_log_likelihood(X, center_list=param[0],
                                     cov_list=param[1], p_k=param[2])







if __name__ == '__main__':
    import pandas as pd
    from util.common import metric
    df = pd.read_csv('/home/tadeze/projects/missingvalue/datasets/anomaly/yeast/fullsamples/yeast_1.csv')
    train_data = df.ix[:, 1:].as_matrix().astype(np.float64)
    # train_lbl = df.ix[:,0] #
    train_lbl = map(int, df.ix[:, 0] == "anomaly")
    gmms = GaussianMixture(n_components=3)
    gmms.fit(train_data)
    score = -gmms.score_samples(train_data)
    print gmms.get_params(False)
    print len(score)
    print metric(train_lbl, score)
    from pypr.clustering import gmm

    cen_lst, cov_lst, p_k, logL = gmm.em_gm(train_data,max_iter=100,K=3)
    score = [-gmm.gm_log_likelihood(train_data[i,:],center_list=cen_lst, cov_list=cov_lst, p_k=p_k)
             for i in range(0,train_data.shape[0])]
    #print score
    print metric(train_lbl, score)

    # Marginalize the
    #m_cen_lst, m_cov_lst, m_p_k
    featur_inc = [0,4]
    marginalize = gmm.marg_dist(featur_inc,cen_lst,cov_lst,p_k)
    score2 = score = [-gmm.gm_log_likelihood(train_data[i,featur_inc],center_list= marginalize[0], cov_list= marginalize[1],
                                             p_k= marginalize[2])
             for i in range(0,train_data.shape[0])]
    print metric(train_lbl, score2)
    ## EGMM
    egmm = GMM(replica=5, comp=(3,4))
    #egmm.replica  =50
    egmm.train(train_data)
    print egmm.parameters
    # gmx = GaussianMixture()
    # print gmm.score(train_data[0,:].reshape([1,train_data.shape[1]])), score[0]
    # param = gmm.get_params(False)
    # print len(param)
    # print len(gmm.means_)
    # print gmm.precisions_[0]
    # gmx.means_ = gmm.means_
    # gmx.precisions_ = gmm.precisions_
    # #gmx.set_params(param)
    # print gmx.score(train_data[0, :].reshape([1, train_data.shape[1]])), score[0]
