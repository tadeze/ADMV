import pyad as pft
import pandas as pd
import numpy as np
import argparse
import os
import pyloda
from pyloda.missing_value_loda import *
import numpy as np
import sklearn.metrics as mt


## MISSING VALUE FLAGES
MISSING_VALUE = -9999.0
missing_value_per = 0.1


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


def train_forest(train_x, check_mv=False, tree_size=1):
    ff = pft.IsolationForest(train_x, ntree=100 * tree_size, nsample=512)
    return ff


def score(test_x, cmv=False):
    sc_gt = ff.score(test_x, cmv)
    return sc_gt


def inject_missing_value(data, num_missing_attribute, alpha,
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


def miss_features(train_x, label, file_name, miss_column):
    # Train the forest
    ensemble_size = 4  # run upto 4 times
    result = pd.DataFrame()
    alpha = 0.1
    input_d = train_x.shape[1]
    miss_prop = np.arange(0.0, 0.35, 0.05)
    fraction_missing_features = int(np.ceil(len(miss_column) * 0.8))

    for en_size in range(1, 5):
        ff = train_forest(train_x, True, en_size)
        for alpha in miss_prop:
            for num_missing in range(0, fraction_missing_features):
                test_x = train_x.copy()
                # print "{0},{1}", num_missing, train_x.shape[1]
                inject_missing_value(test_x, num_missing, alpha, miss_column)
                # Check the value.
                ms_score = ff.score(test_x, False)
                mt = metric(label, ms_score)
                mt_cmv = metric(label, ff.score(test_x, True))
                # print "For ", [num_missing] + mt + [alpha]
                result = result.append(
                    pd.Series([alpha] + [num_missing /
                                         float(len(miss_column))] + mt + mt_cmv + [en_size]),
                    ignore_index=True)
    result.rename(columns={0: "anom_prop", 1: "num_miss_features",
                           2: "auc", 3: "ap", 4: "auc_cm", 5: "ap_cm", 6: "ensemble_size"},
                  inplace=True)
    return result


def lod_miss_features(train_x, label, file_name, miss_column):
    # Train the forest
    ensemble_size = 4  # run upto 4 times
    result = pd.DataFrame()
    #alpha = 0.1
    input_d = train_x.shape[1]
    miss_prop = np.arange(0.0, 0.35, 0.05)
    fraction_missing_features = int(np.ceil(len(miss_column) * 0.8))
    for en_size in range(1, 5):
        pvh = loda_train(train_x, maxk=100 * en_size)
        for alpha in miss_prop:
            for num_missing in range(0, fraction_missing_features):
                test_x = train_x.copy()
                # print "{0},{1}", num_missing, train_x.shape[1]
                inject_missing_value(test_x, num_missing, alpha, miss_column)

                ms_score = loda_score(test_x, pvh=pvh, check_miss=False).nll
                mt = metric(label, ms_score)
                mt_cmv = metric(label,
                                loda_score(test_x, pvh=pvh, check_miss=True).nll)
                # print "For ", [num_missing] + mt + [alpha]
                result = result.append(
                    pd.Series([alpha] + [num_missing /
                                         float(len(miss_column))] + mt + mt_cmv + [en_size]),
                    ignore_index=True)

    result.rename(columns={0: "anom_prop", 1: "num_miss_features",
                           2: "auc", 3: "ap", 4: "auc_cm", 5: "ap_cm", 6: "ensemble_size"},
                  inplace=True)
    return result


def chek_anomaly_ranking(train_data, train_lbl, input_name, algorithm="loda"):
    """
    Check ranking of anomaly point when applying missing values. 
    """
    fraction_missing_features = int(np.ceil(train_x.shape[1] * 0.8))
    result = pd.DataFrame()
    input_d = train_x.shape[1]

    if algorithm == "loda":
        pvvh = loda_train(train_data, maxk=100)

    else:
        # do for iforest for now.
        ff = train_forest(train_data)

    for num_missing in range(0, fraction_missing_features):
        test_x = train_x.copy()
        # print "{0},{1}", num_missing, train_x.shape[1]
        inject_missing_value(test_x, num_missing, alpha)

        ms_score = loda_score(test_x, pvh=pvh, check_miss=False).nll
        mt = metric(label, ms_score)
        mt_cmv = metric(label,
                        loda_score(test_x, pvh=pvh, check_miss=True).nll)
        # print "For ", [num_missing] + mt + [alpha]
        result = result.append(
            pd.Series([alpha] + [num_missing /
                                 float(input_d)] + mt + mt_cmv),
            ignore_index=True)

    result.rename(
        columns={0: "anom_prop", 1: "num_miss_features",
                 2: "auc", 3: "ap", 4: "auc_cm", 5: "ap_cm"},
        inplace=True)


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

    # print train_lbl
    if args.algorithm == "loda":
        result = lod_miss_features(
            train_data, train_lbl, input_name, miss_colmn)
    else:
        result = miss_features(train_data, train_lbl, input_name, miss_colmn)
    # for it in range(1, int(args.iteration)):
    #     result = pd.concat([result, miss_features(
    #         train_data, train_lbl, os.path.basename(args.input.name))])
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
    main()
