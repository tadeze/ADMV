import os
from pandas import read_csv
from util.common import *
tempresult = "../tempresult"
class Egmm:
    def __init__(self, egmm_path="~/adams/kddexperiment/missingdata/algorithm/egmm"):
        self.gmm_path =  egmm_path

    def train(self, file_name, dims, model_output, score_out, clusterlist="3,4,5",ensemble=2, replicates=15, percentile=0.85,
              block_size=15000, skip_cols=1):

        score_out = os.path.join(tempresult, score_out)
        model_output = os.path.join(tempresult, model_output)
        command = self.gmm_path+" -file {0:s} -dims {1:d} -clusterlist {2:s} -ensemble {3:d} -replicates {4:d} -percentile {5:f}" \
                  " -ignoretiny -incremental -blocksize {6:d} -savemodel -m {7:s} -o {8:s} -skipleftcols {9:d}".\
            format(file_name, dims, clusterlist, ensemble, replicates, percentile, block_size, model_output, score_out, skip_cols)

        #print command
        os.system(command)
    def score(self, test_x, dims, model_input, score_out):
        num = np.random.randint(0,99999)
        file_name = os.path.join(tempresult, os.path.basename(model_input).split('.csv')[0]+"_texmp.txt")
        #print file_name
        if os.path.exists(file_name):
            os.remove(file_name)
        header = ",".join([ "V"+str(i) for i in range(0,test_x.shape[1])])
        np.savetxt(file_name,test_x, delimiter=",",header=header)
        return self.score_file(file_name, dims, model_input, score_out, skip_cols=0)
    def score_file(self, file_name, dims, model_input, score_out, clusterlist="3,4,5",ensemble=2, replicates=15, percentile=0.85,
              block_size=15000,skip_cols=1):
        model_input = os.path.join(tempresult, model_input)
        score_out = os.path.join(tempresult, score_out)
        command = self.gmm_path+" -file {0:s} -dims {1:d} -clusterlist {2:s} -ensemble {3:d} -replicates {4:d} -percentile {5:f}" \
                  " -ignoretiny -incremental -blocksize {6:d} -loadmodel -m {7:s} -missmarg -o {8:s} -skipleftcols {9:d}". \
            format(file_name, dims, clusterlist, ensemble, replicates, percentile, block_size, model_input, score_out, skip_cols)
        #print command

        os.system(command)

        nlscore = read_csv(score_out)["nlscore"].as_matrix()
        return nlscore

    #return read_csv(score_out)["score"].as_matrix()

if __name__ == '__main__':
    from util.common import metric
    file_name = "/home/tadeze/projects/missingvalue/datasets/anomaly/mammography/fullsamples/mammography_1.csv"#
    test_file = "/home/tadeze/projects/research/gmmtest/mammo_miss.csv"
    # yeast/fullsamples/yeast_1.csv"
    #file_name = "../yeast_1.csv"
    #file_name ="/nfs/guille/bugid/adams/ifTadesse/kddexperiment/dataset/abalone_benchmark_0709.csv"
    gmm = Egmm("./egmm")
    d = 6
    skip_cols = 1


    #gmm.train(file_name,d,"yeast.mdl","trainout.csv",skip_cols=skip_cols)
    score = gmm.score_file(file_name, d, "yeast.mdl", "score_out.csv", skip_cols=skip_cols)
    score_miss = gmm.score_file(test_file, d, "yeast.mdl", "miss_score_out.csv", skip_cols=skip_cols)
    df = read_csv(file_name)
    ground_score = read_csv('../tempresult/trainout.csv')
    #print df.head(5)
    train_lbl = map(int, df.ix[:, 0] == "anomaly")
    #print train_lbl
    ground_score = ground_score.sort_values(["id"])
    print metric(train_lbl, score)
    print metric(train_lbl, ground_score["score"])


    score_miss = gmm.score_file(test_file, d, "yeast.mdl", "miss_score_out.csv", skip_cols=skip_cols)
    print metric(train_lbl, score_miss)

    #train_x = df.ix[:,6:].as_matrix().astype(np.float64)
    #train_x[0,6] = np.nan
    # scored = gmm.score(train_x, 7, "egmm_model/shuttle.mdl", "egmm_model/score_out.csv")
    # print metric(train_lbl, scored)
    #
    # import algorithm.pyad as pyft
    # ff = pyft.IsolationForest()
    # ff.train(train_x)
    # score = ff.score(train_x)
    # print metric(train_lbl, score)