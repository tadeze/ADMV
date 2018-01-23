import pandas as pd
import argparse
import os
from util.common import *
from missvalueinjector import ADDetector, MissingValueInjector



## Set of experiments

# from algorithm.egmm import Egmm
# def egmm_miss_features(file_name, label, dims, skip_left=1):
#     egmm = Egmm()
#     egmm.train(file_name, dims=dims, skip_cols=skip_left)
#     score = egmm.score(file_name,dims=dims,skip_cols= skip_left)

def algo_miss_features(train_x, label, miss_column, algorithm, file_name, label_field=0):
         # Train the forest
    ensemble_size = 2  # run upto 4 times
    result = pd.DataFrame()
    miss_prop = np.arange(0, 1.1, 0.1)
    d = len(miss_column)
    fraction_missing_features = int(np.ceil(d /np.sqrt(d)))
    ad_detector = ADDetector(alg_type=algorithm, label=label_field)
    mvi_object = MissingValueInjector()
    for en_size in range(1, ensemble_size):

        ad_detector.train(train_x, ensemble_size=en_size, file_name=file_name)
        for alpha in miss_prop:
            for num_missing in range(1, fraction_missing_features):

                #print alpha, num_missing
                test_x = train_x.copy()
                mvi_object.inject_missing_value(test_x, num_missing, alpha, miss_column)
                if algorithm.upper()!='BIFOR':

                    # Check with missing values.
                    if num_missing*alpha>0:
                        # Check the value.
                        ms_score = ad_detector.score(mvi_object.impute_value(test_x.copy(), "SimpleFill"), False)
                        mt = metric(label, ms_score)
                        mt_impute = metric(label, ad_detector.score(mvi_object.impute_value(test_x.copy(),method="MICE"),False)) #impute
                    else:
                        ms_score = ad_detector.score(test_x, False)
                        mt = metric(label, ms_score)
                        mt_impute = mt
                else:
                    mt=mt_impute = [0.0,0.0]  # Just to reduce computation, only run reduced approach when BIFOR is used.
                mt_reduced = metric(label, ad_detector.score(test_x, True)) #Bagging approach
                result = result.append(
                    pd.Series([alpha] + [num_missing /
                                                    float(len(miss_column))] + [mt[0]] + [mt_reduced[0]] +[mt_impute[0]]
                              +[en_size] + [algorithm]),
                    ignore_index=True)

    result.rename(columns={0: "miss_prop", 1: "miss_features_prop",
                           2: "auc_mean_impute",  3: "auc_reduced", 4: "auc_MICE_impute", 5: "ensemble_size", 6:"algorithm"},
                  inplace=True)
    return result

def random_miss_prop(train_x, label,  miss_column, algorithm, file_name):
    """
    Assume all column of train_x can miss.
    :param train_x:
    :param train_y:
    :param algorithm:
    :return:
    """
    miss_prop = np.arange(0.0, 0.45, 0.05)
    result = pd.DataFrame()
    fraction_missing_features = int(np.ceil(len(miss_column) * 0.8))
    ad_detector = ADDetector(alg_type=algorithm)
    mvi_object = MissingValueInjector()
    num_missing = 0
    ad_detector.train(train_x, ensemble_size=1)
    for alpha in miss_prop:
        # print alpha, num_missing
        test_x = train_x.copy()
        #alpha = 0.15
        if alpha > 0:
            num_missing,_ = mvi_object.inject_missing_in_random_cell(test_x, alpha)
            # impute value and perform detection.
            mt_impute = metric(label, ad_detector.score(mvi_object.impute_value(test_x.copy(),method="MICE"), check_miss=False))
            ms_score = ad_detector.score(mvi_object.impute_value(test_x.copy(), method="SimpleFill"), check_miss=False)
            mt_raw = metric(label, ms_score)
        else:
            ms_score = ad_detector.score(test_x, check_miss=False)
            mt_raw = metric(label, ms_score)
            mt_impute = mt_raw  # just assign 0 value when imputation is not applicable.

        reduced_score = ad_detector.score(test_x, True)

            #print test_x[nanv,:]
        mt_reduced = metric(label, reduced_score)
        # print "For ", [num_missing] + mt + [alpha]

        result = result.append(
            pd.Series([alpha] + [num_missing ] + [mt_raw[0]] + [mt_reduced[0]] + [mt_impute[0]]
                      + [algorithm]),
            ignore_index=True)
        #print result
    result.rename(columns={0: "miss_prop", 1: "num_max_miss_features",
                           2: "auc", 3: "auc_reduced", 4: "auc_impute", 5: "algorithm"},
                  inplace=True)
    return result

def miss_proportions_exp(train_x, label, miss_column, algorithm):
    ensemble_size = 2  # run upto 4 times
    result = pd.DataFrame()
    miss_prop = np.arange(0.05, 0.45, 0.05)
    fraction_missing_features = int(np.ceil(len(miss_column) * 0.8))
    ad_detector = ADDetector(alg_type=algorithm)
    mvi_object = MissingValueInjector()
    for en_size in range(1, ensemble_size):
        ad_detector.train(train_x, ensemble_size=en_size)
        for alpha in miss_prop:
            for num_missing in range(0, fraction_missing_features):
                # print alpha, num_missing
                test_x = train_x.copy()
                ms_score = ad_detector.score(test_x, False)
                mt_raw = metric(label, ms_score)

                if num_missing>0:
                    mvi_object.inject_missing_value(test_x, num_missing, alpha, miss_column)
                    # impute value and perform detection.
                    mt_impute = metric(label, ad_detector.score(mvi_object.impute_value(test_x), False))
                else:
                    mt_impute =[0,0] # just assign 0 value when imputation is not applicable.
                mt_reduced = metric(label, ad_detector.score(test_x.copy(), True))
                    # print "For ", [num_missing] + mt + [alpha]

                result = result.append(
                    pd.Series([alpha] + [num_missing /
                                         float(len(miss_column))] + [mt_raw[0]] + [mt_reduced[0]] + [mt_impute[0]]
                              + [en_size] + [algorithm]),
                    ignore_index=True)

    result.rename(columns={0: "anom_prop", 1: "num_max_miss_features",
                           2: "auc", 3: "auc_reduced", 4: "auc_impute", 5: "ensemble_size", 6: "algorithm"},
                  inplace=True)
    return result




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
    parser.add_argument('-t', '--type', help="Experiment type.")
    parser.add_argument('-o', '--outputdir', help="Output directory location")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.outputdir is None:
            output_dir = "/scratch/cluster-share/zemicheal/missingdata/kddexp"
    else:
        output_dir = args.outputdir


    column = map(int, args.column.split('-'))
    if len(column)>1:
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
        miss_colmn = map(int, args.missing.split(','))
    else:
        # re-adjust the missing column index.
        miss_colmn = range(0, len(miss_colmn))

    # print df.ix[1:29,:]

    input_name = os.path.basename(args.input.name)
    if args.type=='cell':
        result = random_miss_prop(train_data, train_lbl, miss_colmn, str(args.algorithm).upper(),
                                  file_name=args.input.name)
    else:
        result = algo_miss_features(train_data, train_lbl, miss_colmn, str(args.algorithm).upper(),
                                    file_name=args.input.name, label_field=args.label)


        # # print train_lbl
    # if args.algorithm == "loda":
    #     result = lod_miss_features(
    #         train_data, train_lbl, input_name, miss_colmn)
    # else:
    #     result = ifor_miss_features(
    #         train_data, train_lbl, input_name, miss_colmn)
    #for it in range(1, int(args.iteration)):
     #    result = pd.concat([result, miss_features(
     #        train_data, train_lbl, os.path.basename(args.input.name))])
    #result = None

    dir_name = os.path.join(output_dir, args.algorithm,
                            os.path.splitext(input_name)[0])
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    experiment_type = args.type
    result.to_csv(dir_name + '/'+experiment_type+'_results_iter_' +
                  str(args.iteration) + input_name)

    #grp = result.group_by(['anom_prop','num_miss_features']).agg([np.mean])

if __name__ == '__main__':
    #test()
    main()
