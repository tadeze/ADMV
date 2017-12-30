import pandas as pd
import argparse
from common import *
import os 

def miss_values(explanation_file, score_file, orgfile):
    expdata = pd.read_csv(explanation_file)
    score = pd.read_csv(score_file).sort_values("id")
    org_data = pd.read_csv(orgfile)
    label = map(int, org_data.ix[:, 0] == "anomaly")
    ncol = org_data.shape[1] - 1
    miss_features = expdata.ix[:, 1]
    count_miss = map(lambda x: str(x).count("0"), miss_features)
    #maxmiss = max(count_miss) + 1
    marg_score = expdata.ix[:, 2:]
    nrow = marg_score.shape[1]
    index_miss_att = {}

    for i, x in enumerate(count_miss):
        if index_miss_att.has_key(x):
            index_miss_att[x].append(i)
        else:
            index_miss_att[x] = [i]

    def inject_missing_value(num_missing, num_miss_features):

        miss_index = index_miss_att[num_miss_features]
        if miss_index is None:
            return None
        miss_row_score = {}
        # chose random proportion as missing values
        prop_miss = np.random.choice(
            range(0, nrow + 1), num_missing, replace=False)
        for ix in prop_miss:
            rowix = np.random.choice(miss_index, 1, replace=False)[0]
            miss_row_score[ix] = marg_score.ix[rowix, ix]
        return miss_row_score

    result = pd.DataFrame()
    # alpha = 0.1
    miss_prop = np.arange(0.05, 0.35, 0.05)
    fraction_missing_features = int(np.ceil(ncol * 0.8))

    for alpha in miss_prop:
        for num_missing in range(1, fraction_missing_features):
            current_score = score.copy()
            num_miss_features = int(np.ceil(alpha * ncol))
            injected_score = inject_missing_value(
                num_missing, num_miss_features)
            if injected_score is None:
                continue
            # joint to the common scores
            current_score.ix[injected_score.keys(), 'score'] = map(
                lambda x: x * -1, injected_score.values())
            mtx = metric(label, current_score.ix[:, 'score'])
            # print "For ", [num_missing] + mt + [alpha]
            result = result.append(
                pd.Series([alpha] + [num_missing] + mtx), ignore_index=True)

    result.rename(columns={0: "anom_prop", 1: "num_miss_features",
                           2: "auc", 3: "ap", 4: "auc_cm", 5: "ap_cm",
                           6: "ensemble_size"},
                  inplace=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="EGMM ussage")

    parser.add_argument('-e', '--explanation', type=argparse.FileType('r'),
                        help="Required input file", required=True)
    parser.add_argument('-s', '--score', type=argparse.FileType('r'),
                        help="Required input file", required=True)
    parser.add_argument('-i', '--input', type=argparse.FileType('r'),
                        help="Required input file", required=True)
    parser.add_argument('-n','--iter')
    # parser.add_argument('-c', '--column', help="Column hypheneted")
    # parser.add_argument('-m', '--missing', help="Missing injection columns")
    # parser.add_argument('-l', '--label', help="Ground flag label")
    # parser.add_argument('-n', '--iteration', help="Number of iterations")
    # parser.add_argument('-g', '--algorithm', help="Type of algorithm to use")
    # parser.add_argument('-e', '--ensemble', help='Ensemble size. Defualt 1')
    args = parser.parse_args()
    # "eggmresult/allDensity_shuttle1.csv"
    explanation_file = args.explanation.name
    score_file = args.score.name  # "eggmresult/out_shuttle1.csv"
    # 'anomaly/shuttle_1v23567/fullsamples/shuttle_1v23567_1.csv'
    org_file = args.input.name
    dt = miss_values(explanation_file, score_file, org_file)
    output_dir = "./egmmresult/result"
    dir_name = os.path.join(output_dir, args.iter+os.path.basename(org_file))
    dt.to_csv(dir_name)


if __name__ == '__main__':
    main()
