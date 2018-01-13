from algorithm.loda.loda import Loda
from algorithm import BaggedIForest
from util.common import *
import fancyimpute as fi
import pyad as pft


class ADDetector:
    def __init__(self, alg_type="IFOR"):
        self.alg_type = alg_type


    def train(self, x_train, ensemble_size=1):
        """
        Train algorithm, based on its type.
        """
        if self.alg_type == "BIFOR":
            self.ad_model = BaggedIForest(ntree=100*ensemble_size)
        elif self.alg_type == "IFOR":
            self.ad_model = pft.IsolationForest(ntree=100*ensemble_size)
        else:
            self.ad_model = Loda(maxk=100*ensemble_size)
        self.ad_model.train(x_train)
        #self.increase_ensemble(ensemble_size)
        #self.ad_model.train(x_train)
    def increase_ensemble(self, k):
        if self.alg_type=="IFOR" or self.alg_type=="BIFOR":
            self.ad_model.ntree = k*self.ad_model.ntree
        else:
            self.ad_model.maxk = self.ad_model.maxk*k
        ## call train.
    def score(self, x_test, check_miss=True):
        # Replace the np.nan with large number.
        #if check_miss:
        x_test[np.isnan(x_test)] = MISSING_VALUE
        return self.ad_model.score(x_test, check_miss)


class MissingValueInjector(object):
    def __init__(self,):
        pass
        # configure common injection mechanism


    def impute_value(self, df, method="MICE"):
        """
        Impute using MICE
        """
        if method == "MICE":
            return fi.MICE(verbose=False).complete(df)
        elif method == "KNN":
            return fi.KNN(k=4, verbose=False).complete(df)
        else:
            return fi.SimpleFill().complete(df)


    def inject_missing_value(self, data, num_missing_attribute, alpha,
                             miss_att_list):

        missing_amount = int(np.ceil(data.shape[0] * alpha))
        missing_index = np.random.choice(
            data.shape[0], missing_amount, replace=False)
        miss_att_list_len = len(miss_att_list)
        # print miss_att_list
        # Insert missing value at completley random.


        for index in missing_index:
            miss_att = np.random.choice(
                miss_att_list, num_missing_attribute, replace=False)
            if len(miss_att) > 0:
                data[index, miss_att] = NA #MISSING_VALUE
    def inject_missing_in_random_cell(self, data,  alpha):

        """
        :param data:
        :param num_missing_attribute: Number of missing attributes
        :param alpha: missing amount proportion
        :param miss_att_list: missing attribute list
        :return:
        """
        assert isinstance(data, np.ndarray)

        missing_amount = int(np.ceil(data.size*alpha))
        missing_index = np.random.choice(data.size, missing_amount,False)
        for index in missing_index:
            data.itemset(index, NA)
        affected_rows = np.argwhere(np.isnan(data))
        return max(np.isnan(data).sum(1)),affected_rows