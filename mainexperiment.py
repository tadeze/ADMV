import pandas as pd
import numpy as np
import argparse
import os
from algorithm.loda.loda import Loda
from algorithm import BaggedIForest
from util.common import *
import fancyimpute as fi

class ADDetector:
    def __init__(self, alg_type="iFOR"):
        self.alg_type = alg_type
        if self.alg_type == "IFOR":
            self.ad_model = BaggedIForest(ntree=100)
        else:
            self.ad_model = Loda(maxk=100)

    def train(self, x_train, ensemble_size=1):
        """
        Train algorithm, based on its type. 
        """
        self.increase_ensemble(ensemble_size)
        self.ad_model.train(x_train)
    def increase_ensemble(self, k):
        if self.alg_type=="IFOR":
            self.ad_model.ntree = k*self.ad_model.ntree
        else:
            self.ad_model.maxk = self.ad_model.maxk*k
        ## call train.
    def score(self, x_test):
        return self.ad_model.score(x_test)


class MissingValueInjector(object):
    def __init__(self,):
        pass
        # configure common injection mechanism


    def impute_value(self, df, method="MICE"):
        """
        Impute using MICE
        """
        if method=="MICE":
            return fi.MICE().complete(df)
        elif method=="KNN":
            return fi.KNN(k=4).complete(df)
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
                data[index, miss_att] = MISSING_VALUE


def algo_miss_features(train_x, label, file_name, miss_column, algorithm):
         # Train the forest
    ensemble_size = 2  # run upto 4 times
    result = pd.DataFrame()
    alpha = 0.1
    input_d = train_x.shape[1]
    miss_prop = np.arange(0.0, 0.35, 0.05)
    fraction_missing_features = int(np.ceil(len(miss_column) * 0.8))
    ad_detector = ADDetector(alg_type=algorithm)
    mvi_object = MissingValueInjector()
    for en_size in range(1, ensemble_size):
        ad_detector.train(train_x, ensemble_size=en_size)
        for alpha in miss_prop:
            for num_missing in range(0, fraction_missing_features):
                alpha = 0.1
                num_missing = 2
                test_x = train_x.copy()
                # print "{0},{1}", num_missing, train_x.shape[1]
                mvi_object.inject_missing_value(test_x, num_missing, alpha, miss_column)
                # Check the value.
                ms_score = ad_detector.score(train_x)
                mt = metric(label, ms_score)
                # Check with missing values.
                mt_reduced = metric(label, ad_detector.score(test_x))
                mt_impute = metric(label, ad_detector.score(mvi_object.impute_value(test_x)))
                # print "For ", [num_missing] + mt + [alpha]
                result = result.append(
                    pd.Series([alpha] + [num_missing /
                                                    float(len(miss_column))] + [mt[0]] + [mt_reduced[0]] +[mt_impute[0]]+ [en_size]),
                    ignore_index=True)

    result.rename(columns={0: "anom_prop", 1: "num_miss_features",
                           2: "auc",  3: "auc_reduced", 4: "auc_impute", 5: "ensemble_size"},
                  inplace=True)
    return result




# def ifor_miss_features(train_x, label, file_name, miss_column):
#     # Train the forest
#     ensemble_size = 4  # run upto 4 times
#     result = pd.DataFrame()
#     alpha = 0.1
#     input_d = train_x.shape[1]
#     miss_prop = np.arange(0.0, 0.35, 0.05)
#     fraction_missing_features = int(np.ceil(len(miss_column) * 0.8))
#
#     for en_size in range(1, 5):
#         ff = train_forest(train_x, True, en_size)
#         for alpha in miss_prop:
#             for num_missing in range(0, fraction_missing_features):
#                 test_x = train_x.copy()
#                 # print "{0},{1}", num_missing, train_x.shape[1]
#                 inject_missing_value(test_x, num_missing, alpha, miss_column)
#                 # Check the value.
#                 ms_score = ff.score(test_x, False)
#                 mt = metric(label, ms_score)
#                 mt_cmv = metric(label, ff.score(test_x, True))
#                 # print "For ", [num_missing] + mt + [alpha]
#                 result = result.append(
#                     pd.Series([alpha] + [num_missing /
#                                          float(len(miss_column))] + mt + mt_cmv + [en_size]),
#                     ignore_index=True)
#     result.rename(columns={0: "anom_prop", 1: "num_miss_features",
#                            2: "auc", 3: "ap", 4: "auc_cm", 5: "ap_cm", 6: "ensemble_size"},
#                   inplace=True)
#     return result


# def lod_miss_features(train_x, label, file_name, miss_column):
#     # Train the forest
#     ensemble_size = 4  # run upto 4 times
#     result = pd.DataFrame()
#     #alpha = 0.1
#     input_d = train_x.shape[1]
#     miss_prop = np.arange(0.0, 0.35, 0.05)
#     fraction_missing_features = int(np.ceil(len(miss_column) * 0.8))
#     for en_size in range(1, 5):
#         pvh = loda_train(train_x, maxk=100 * en_size)
#         for alpha in miss_prop:
#             for num_missing in range(0, fraction_missing_features):
#                 test_x = train_x.copy()
#                 # print "{0},{1}", num_missing, train_x.shape[1]
#                 inject_missing_value(test_x, num_missing, alpha, miss_column)
#                 ms_score = loda_score(test_x, pvh=pvh, check_miss=False).nll
#                 mt = metric(label, ms_score)
#                 mt_cmv = metric(label,
#                                 loda_score(test_x, pvh=pvh, check_miss=True).nll)
#                 # print "For ", [num_missing] + mt + [alpha]
#                 result = result.append(
#                     pd.Series([alpha] + [num_missing /
#                                          float(len(miss_column))] + mt + mt_cmv + [en_size]),
#                     ignore_index=True)
#
#     result.rename(columns={0: "anom_prop", 1: "num_miss_features",
#                            2: "auc", 3: "ap", 4: "auc_cm", 5: "ap_cm", 6: "ensemble_size"},
#                   inplace=True)
#     return result

#
# def chek_anomaly_ranking(train_x, train_lbl, input_name, algorithm="loda"):
#     """
#     Check ranking of anomaly point when applying missing values.
#     """
#     fraction_missing_features = int(np.ceil(train_x.shape[1] * 0.8))
#     result = pd.DataFrame()
#     input_d = train_x.shape[1]
#
#     if algorithm == "loda":
#         pvvh = loda_train(train_data, maxk=100)
#
#     else:
#         # do for iforest for now.
#         ff = train_forest(train_data)
#
#     for num_missing in range(0, fraction_missing_features):
#         test_x = train_x.copy()
#         # print "{0},{1}", num_missing, train_x.shape[1]
#         inject_missing_value(test_x, num_missing, alpha)
#
#         ms_score = loda_score(test_x, pvh=pvh, check_miss=False).nll
#         mt = metric(label, ms_score)
#         mt_cmv = metric(label,
#                         loda_score(test_x, pvh=pvh, check_miss=True).nll)
#         # print "For ", [num_missing] + mt + [alpha]
#         result = result.append(
#             pd.Series([alpha] + [num_missing /
#                                  float(input_d)] + mt + mt_cmv),
#             ignore_index=True)
#
#     result.rename(
#         columns={0: "anom_prop", 1: "num_miss_features",
#                  2: "auc", 3: "ap", 4: "auc_cm", 5: "ap_cm"},
#         inplace=True)


def main():
    parser = argparse.ArgumentParser(description="iForest usage switches")
    parser.add_argument('-i', '--input', type=argparse.FileType('r'),
                        help="Required input file", required=True)
    parser.add_argument('-c', '--column', help="Column hypheneted")
    parser.add_argument('-m', '--missing', help="Missing injection columns")
    parser.add_argument('-l', '--label', help="Ground flag label")
    parser.add_argument('-n', '--iteration', help="Number of iterations")
    parser.add_argument('-g', '--algorithm', help="Type of algorithm to use")
    parser.add_argument('-e', '--ensemble', help='Ensemble size. Defualt 1')
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    column = map(int, args.column.split('-'))

    miss_colmn = range(column[0], (column[1] + 1))

    lbl = int(args.label)
    if type(df.ix[3, lbl]) is str:
        train_lbl = map(int, df.ix[:, lbl] == "anomaly")
    else:
        train_lbl = df.ix[:, lbl]

    train_data = df.ix[:, miss_colmn].as_matrix()
    if args.missing is not None:
        miss_colmn = map(int, args.missing.split(','))
    else:
        # re-adjust the missing column index.
        miss_colmn = range(0, len(miss_colmn))

    # print df.ix[1:29,:]

    input_name = os.path.basename(args.input.name)
    #
    # # print train_lbl
    # if args.algorithm == "loda":
    #     result = lod_miss_features(
    #         train_data, train_lbl, input_name, miss_colmn)
    # else:
    #     result = ifor_miss_features(
    #         train_data, train_lbl, input_name, miss_colmn)
    # for it in range(1, int(args.iteration)):
    #     result = pd.concat([result, miss_features(
    #         train_data, train_lbl, os.path.basename(args.input.name))])
    result = None
    output_dir = "/scratch/cluster-share/zemicheal/missingdata/ensemble"
    #output_dir = "tempdir/"
    dir_name = os.path.join(output_dir, args.algorithm,
                            os.path.splitext(input_name)[0])
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    result.to_csv(dir_name + '/results_iter_' +
                  str(args.iteration) + input_name)

    #grp = result.group_by(['anom_prop','num_miss_features']).agg([np.mean])
if __name__ == '__main__':
    #main()
     df = pd.read_csv("yeast_1.csv")
     train_data = df.ix[:,1:].as_matrix()
     train_lbl = map(int, df.ix[:, 0] == "anomaly")
     miss_colmn = range(0,4)
     result = algo_miss_features(
         train_data, train_lbl, "yeast.csv", miss_colmn,'IFOR')
     print result
