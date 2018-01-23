import argparse
import pandas as pd
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
        miss_colmn = map(int, args.missing.split(','))
    else:
        # re-adjust the missing column index.
        miss_colmn = range(0, len(miss_colmn))

    # print df.ix[1:29,:]

    input_name = os.path.basename(args.input.name)
    if args.type == 'cell':
        result = random_miss_prop(train_data, train_lbl, miss_colmn, str(args.algorithm).upper(),
                                  file_name=args.input.name)
    else:
        result = algo_miss_features(train_data, train_lbl, miss_colmn, str(args.algorithm).upper(),
                                    file_name=args.input.name, label_field=args.label)

