from algorithm.loda.loda import Loda
from algorithm import BaggedIForest
import algorithm.pyad as pft
from algorithm.lof import BaggedLOF
from util.common import *
import fancyimpute as fi
from algorithm import Egmm
import os

class ADDetector:
    """
    Anomaly detector classes. A generic class for training and testing anomaly detectors.
    """
    def __init__(self, alg_type="IFOR", label=0):
        self.alg_type = alg_type.upper()
        self.label = label

    def train(self, x_train, ensemble_size=1, file_name=None):
        """
        Train algorithm, based on its type.
        """
        if self.alg_type == "BIFOR":
            self.ad_model = BaggedIForest(ntree=100*ensemble_size)
        elif self.alg_type == "IFOR":
            self.ad_model = pft.IsolationForest(ntree=100*ensemble_size)
        elif self.alg_type =="LOF":
            self.ad_model = BaggedLOF(num_model=30)
        elif self.alg_type == "LODA":
            self.ad_model = Loda(maxk=100*ensemble_size)
        elif self.alg_type == "EGMM":
            if file_name is None:
                return ValueError("No correct file name given")

            ## Train egmm model
            self.ad_model = Egmm("algorithm/egmm")
            self.dims = x_train.shape[1]

            self.model_output = os.path.basename(file_name)+".mdl"
            self.score_out = os.path.basename(file_name)
            self.ad_model.train(file_name,dims=self.dims,model_output=self.model_output,
                                score_out=self.score_out+".tr", skip_cols=int(self.label)+1)
            return 0
        else:
            return ValueError("Incorrect algorithm name")
        self.ad_model.train(x_train)

    def increase_ensemble(self, k):
        if self.alg_type=="IFOR" or self.alg_type=="BIFOR":
            self.ad_model.ntree = k*self.ad_model.ntree
        else:
            self.ad_model.maxk = self.ad_model.maxk*k
        ## call train.
    def score(self, x_test, check_miss=True):
        """

        :param x_test:
        :param check_miss: If true, check if there is any missing value in the testing data and treat it.
        :return:
        """

        if self.alg_type =="EGMM":
            num = np.random.randint(0,99999)
            return self.ad_model.score(x_test,dims=self.dims, model_input=self.model_output,
                                       score_out=self.score_out+str(num)+".sc")
        return self.ad_model.score(x_test, check_miss)


class MissingValueInjector(object):
    def __init__(self,):
        """
        Missing value inject and imputation model.
        """
        self.__name__ = "Missing value injector"


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

    def inject_miss(self, missing_index, num_missing_attribute, data, miss_att_list):
        for index in missing_index:
            miss_att = np.random.choice(
                miss_att_list, num_missing_attribute, replace=False)
            if len(miss_att) > 0:
                data[index, miss_att] = np.nan  # MISSING_VALUE

    def inject_missing_value(self, data, num_missing_attribute, alpha,
                             miss_att_list=[]):
        """

        :param data: numpy.ndarray nXd data
        :param num_missing_attribute: Number of missing attribtes from d dimension. <d
        :param alpha: fraction of data with missing attributes
        :param miss_att_list: subset of selected features that not be inject a missing value.
        :return:
        """

        missing_amount = int(np.ceil(data.shape[0] * alpha))

        # Incase of decimal deimension,, ceil or fllow it
        floor_percent = lambda d: d - np.floor(d)
        ceil_amount = int(missing_amount*floor_percent(num_missing_attribute)) # on ceil(num_missing_attributes)

        missing_index = np.random.choice(
            data.shape[0], missing_amount, replace=False)
        np.random.shuffle(missing_index)

        self.inject_miss(missing_index[:ceil_amount], int(np.ceil(num_missing_attribute)), data, miss_att_list)
        self.inject_miss(missing_index[ceil_amount:], int(np.floor(num_missing_attribute)), data, miss_att_list)


        return np.where(np.isnan(data))  #Return index of affected cells.
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
            data.itemset(index, np.nan)
        affected_rows = np.argwhere(np.isnan(data))
        return max(np.isnan(data).sum(1)),affected_rows
