# import pyximport; pyximport.install()
# from benchbase import *
import pandas as pd
import argparse
import os
from util.common import *
from missvalueinjector import ADDetector, MissingValueInjector
from joblib import Parallel, delayed
import multiprocessing
import logging
logging.basicConfig(level=logging.DEBUG, filename="test.log")
ALGORITHMS = ["IFOR", "BIFOR", "LODA","EGMM"]


def replace_with_nan(index, df):
    df[index] = np.nan

def algo_parameters(t_id):
    t_id -=1
    fraction_missing_features = np.arange(0,0.9,0.1)  # int(np.ceil(d / np.sqrt(d)))
    miss_prop = np.arange(0, 1.1, 0.1)
    #print miss_prop, fraction_missing_features
    m, f = len(miss_prop), len(fraction_missing_features)
    if t_id >= m*f:
        return ValueError("Value should not be greater than allowed")
    t_miss = t_id % m
    t_frac = t_id / m
    return miss_prop[t_miss], fraction_missing_features[t_frac]


def single_benchmark(train_x, label, miss_column, file_name, label_field, algorithm_list=ALGORITHMS, task_id=1):
    d = len(miss_column)  # size of missing features.
    frac_missing_prop, frac_features = algo_parameters(task_id)
    num_missing_att = int(np.ceil(d*frac_features))
    test_x = train_x.copy()
    miss_index = []
    scores_result = []

    algorithms = {}
    mvi_object = MissingValueInjector()
    x_impute = lambda method: mvi_object.impute_value(test_x, method=method)

    for algo in algorithm_list:
        try:

            algorithms[algo] = ADDetector(alg_type=algo, label=label_field)
            algorithms[algo].train(train_x, ensemble_size=1, file_name=file_name)
        except  Exception as e:
            print "Error from {0:s}".format(algo), e.message
            continue

    def score_algo(test_x, method, score_bool=False):
        scores = []
        local_score =[]
        for algo in algorithms:
            try:
                if (method in ["mean", "MICE"]) and algo == "BIFOR":
                    continue
                local_score = algorithms[algo].score(test_x, score_bool)
                logging.debug(
                    "score {0:d} - {1:s} - {2:s}-{3:s}".format(len(local_score), file_name, str(frac_features),
                                                               str(frac_missing_prop)))
                auc_score = metric(label,local_score)[0]

                scores.append([frac_missing_prop, frac_features, auc_score, algo, method, os.path.basename(file_name)])
            except Exception as e:
                print "Error from {0:s}".format(algo), e.message
                continue

        return scores

    if num_missing_att*frac_missing_prop>0:
        miss_index = mvi_object.inject_missing_value(data=test_x, num_missing_attribute=num_missing_att,
                                                     alpha=frac_missing_prop, miss_att_list=miss_column)
        x_na_mean = x_impute("SimpleFill")
        scores_result += score_algo(x_na_mean,method="mean",score_bool=False)
        replace_with_nan(miss_index, test_x)
        x_na_mice = x_impute("MICE")
        scores_result += score_algo(x_na_mice,method="MICE", score_bool=False) # append the list

    else:
        scores_result += score_algo(test_x,method="NoImpute")

    replace_with_nan(miss_index, test_x)
    scores_result += score_algo(test_x, method="reduced",score_bool=True)
    return scores_result

def benchmarks(train_x, label, miss_column, algorithm, alpha, num_missing):
    test_x = train_x
    miss_index = mvi_object.inject_missing_value(test_x, num_missing, alpha, miss_column)


    if algorithm.upper() != 'BIFOR':

        # Check with missing values.
        if num_missing * alpha > 0:
            # Check the value.
            ms_score = ad_detector.score(mvi_object.impute_value(test_x, "SimpleFill"), False)
            mt = metric(label, ms_score)
            replace_with_nan(miss_index, test_x)
            mt_impute = metric(label, ad_detector.score(mvi_object.impute_value(test_x, method="MICE"),
                                                        False))  # impute
        else:
            replace_with_nan(miss_index, test_x)
            ms_score = ad_detector.score(test_x, False)
            mt = metric(label, ms_score)
            mt_impute = mt
    else:
        mt = mt_impute = [0.0, 0.0]  # Just to reduce computation, only run reduced approach when BIFOR is used.
    replace_with_nan(miss_index, test_x)
    mt_reduced = metric(label, ad_detector.score(test_x, True))  # Bagging approach
    #print mt_reduced
    return    [alpha] + [num_missing /
                             float(len(miss_column))] + [mt[0]] + [mt_reduced[0]] + [mt_impute[0]] \
                        + [algorithm]

def algo_miss_featuresX(train_x, label, miss_column, algorithm, file_name, label_field=0):
    """
    For running locally with threaded process. This is useful, if the job can only run on a single node.
    :param train_x:
    :param label:
    :param miss_column:
    :param algorithm:
    :param file_name:
    :param label_field:
    :return:
    """
    global ad_detector, mvi_object
    # Train the forest
    result = pd.DataFrame()
    miss_prop = np.arange(0, 1.1, 0.1)
    d = len(miss_column)
    fraction_missing_features = int(np.ceil(d * 0.8))  # int(np.ceil(d / np.sqrt(d)))
    ad_detector = ADDetector(alg_type=algorithm, label=label_field)
    mvi_object = MissingValueInjector()
    ad_detector.train(train_x, ensemble_size=1, file_name=file_name)
    num_cores = multiprocessing.cpu_count()
    result = Parallel(n_jobs=num_cores)(delayed(benchmarks, check_pickle=False)
                                        (train_x, label, miss_column, algorithm, alpha, num_miss)
                                        for alpha in miss_prop for num_miss in
                                        range(1, fraction_missing_features))

    result = pd.DataFrame(result)
    result.rename(columns={0: "miss_prop", 1: "miss_features_prop",
                           2: "auc_mean_impute", 3: "auc_reduced", 4: "auc_MICE_impute", 5: "ensemble_size",
                           6: "algorithm"}, inplace=True)

    return result


def algo_miss_features(train_x, label, miss_column, algorithm, file_name, label_field=0):
    # Slower version,, not the latest version to use.
    ensemble_size = 2  # run upto 4 times
    result = pd.DataFrame()
    miss_prop = np.arange(0, 1.1, 0.1)
    d = len(miss_column)
    fraction_missing_features = int(np.ceil(len(miss_column) * 0.8))  # int(np.ceil(d /np.sqrt(d)))
    ad_detector = ADDetector(alg_type=algorithm, label=label_field)
    mvi_object = MissingValueInjector()
    for en_size in range(1, ensemble_size):

        ad_detector.train(train_x, ensemble_size=en_size, file_name=file_name)
        for alpha in miss_prop:
            for num_missing in range(1, fraction_missing_features):

                # print alpha, num_missing
                test_x = train_x.copy()
                mvi_object.inject_missing_value(test_x, num_missing, alpha, miss_column)
                if algorithm.upper() != 'BIFOR':

                    # Check with missing values.
                    if num_missing * alpha > 0:
                        # Check the value.
                        ms_score = ad_detector.score(mvi_object.impute_value(test_x.copy(), "SimpleFill"), False)
                        mt = metric(label, ms_score)
                        mt_impute = metric(label,
                                           ad_detector.score(mvi_object.impute_value(test_x.copy(), method="MICE"),
                                                             False))  # impute
                    else:
                        ms_score = ad_detector.score(test_x, False)
                        mt = metric(label, ms_score)
                        mt_impute = mt
                else:
                    mt = mt_impute = [0.0,
                                      0.0]  # Just to reduce computation, only run reduced approach when BIFOR is used.
                mt_reduced = metric(label, ad_detector.score(test_x, True))  # Bagging approach
                result = result.append(
                    pd.Series([alpha] + [num_missing /
                                         float(len(miss_column))] + [mt[0]] + [mt_reduced[0]] + [mt_impute[0]]
                              + [en_size] + [algorithm]),
                    ignore_index=True)

    result.rename(columns={0: "miss_prop", 1: "miss_features_prop",
                           2: "auc_mean_impute", 3: "auc_reduced", 4: "auc_MICE_impute", 5: "ensemble_size",
                           6: "algorithm"},
                  inplace=True)
    return result


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
    parser.add_argument('-t', '--type', help="Experiment type.")
    parser.add_argument('-o', '--outputdir', help="Output directory location")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.outputdir is None:
        output_dir = "/scratch/cluster-share/zemicheal/missingdata/kddexp"
    else:
        output_dir = args.outputdir

    column = map(int, args.column.split('-'))
    if len(column) > 1:
        miss_colmn = range(column[0], (column[1] + 1))
        train_data = df.ix[:, miss_colmn].as_matrix().astype(np.float64)
    else:
        train_data = df.ix[:, column[0]:].as_matrix().astype(np.float64)
        miss_colmn = range(train_data.shape[1])
    lbl = int(args.label)

    if type(df.ix[3, lbl]) is str:
        train_lbl = map(int, df.ix[:, lbl] == "anomaly")
    else:
        train_lbl = df.ix[:, lbl]

    if args.missing is not None:
        print args.missing
        miss_colmn = map(int, args.missing.split(','))
    else:
        # re-adjust the missing column index.
        miss_colmn = range(0, len(miss_colmn))

    # print df.ix[1:29,:]

    input_name = os.path.basename(args.input.name)
    if args.type == 'cell':
        result = algo_miss_features(train_data, train_lbl, miss_colmn, str(args.algorithm).upper(),
                                  file_name=args.input.name)
    elif args.type=="features":
        if str(args.algorithm).upper() == "EGMM":
            result = algo_miss_features(train_data, train_lbl, miss_colmn, str(args.algorithm).upper(),
                                        file_name=args.input.name, label_field=args.label)
        else:
            result = algo_miss_featuresX(train_data, train_lbl, miss_colmn, str(args.algorithm).upper(),
                                         file_name=args.input.name, label_field=args.label)
    else:
        if args.algorithm is not None:
            algo_list = args.algorithm.split(',')
        else:
            algo_list = ALGORITHMS
        result = single_benchmark(train_x=train_data, label=train_lbl, miss_column=miss_colmn, file_name=args.input.name,
                                  label_field=lbl, task_id=int(args.iteration),algorithm_list=algo_list)
        result = pd.DataFrame(result)
    result.rename(columns={0: "miss_prop", 1: "miss_features_prop",
                           2: "auc", 3: "algorithm", 4: "method", 5: "bench_ame"
                           }, inplace=True)

    dir_name = os.path.join(output_dir,
                            os.path.splitext(input_name)[0])
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    experiment_type = args.type
    result.to_csv(dir_name + '/' + experiment_type + '_results_iter_' +
                  str(args.iteration) + input_name)

if __name__ == '__main__':
    # test()
    main()
