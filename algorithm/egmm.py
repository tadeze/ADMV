import os
from pandas import read_csv
import numpy as np
from util.common import *
## Train
## - includes, train the model and save it.
## - score:
## - includes,, load the model & generate score from the model.
class Egmm:
    def __init__(self):
        self.gmm_path = "~/adams/kddexperiment/missingdata/algorithm/egmm"

    def train(self, file_name, dims, model_output, score_out, clusterlist="3,4,5",ensemble=2, replicates=15, percentile=0.85,
              block_size=15000, skip_cols=1):

        command = self.gmm_path+" -file {0:s} -dims {1:d} -clusterlist {2:s} -ensemble {3:d} -replicates {4:d} -percentile {5:f}" \
                  " -ignoretiny -incremental -blocksize {6:d} -savemodel -m {7:s} -o {8:s} -skipleftcols {9:d}".\
            format(file_name, dims, clusterlist, ensemble, replicates, percentile, block_size, model_output, score_out, skip_cols)

        #print command
        os.system(command)
    def score(self, test_x, dims, model_input, score_out):
        file_name = os.path.join(tempresult,"texmp.txt")
        print file_name
        if os.path.exists(file_name):
            os.remove(file_name)
        header = ",".join([ "V"+str(i) for i in range(0,test_x.shape[1])])
        np.savetxt(file_name,test_x, delimiter=",",header=header)
        return self.score_file(file_name, dims, model_input, score_out, skip_cols=0)
    def score_file(self, file_name, dims, model_input, score_out, clusterlist="3,4,5",ensemble=2, replicates=15, percentile=0.85,
              block_size=15000,skip_cols=1):
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
    #file_name = "/home/tadeze/projects/missingvalue/datasets/anomaly/yeast/fullsamples/yeast_1.csv"
    #file_name = "../yeast_1.csv"
    file_name ="/nfs/guille/bugid/adams/ifTadesse/kddexperiment/dataset/abalone_benchmark_0709.csv"
    gmm = Egmm()
    gmm.train(file_name,7,"egmm_model/shuttle.mdl","egmm_model/score.csv",skip_cols=6)
    score = gmm.score_file(file_name,7,"egmm_model/shuttle.mdl", "egmm_model/score_out.csv", skip_cols=6)
    df = read_csv(file_name)
    print df.head(5)
    train_lbl = map(int, df.ix[:, 5] == "anomaly")
    print train_lbl
    print metric(train_lbl, score)
    train_x = df.ix[:,6:].as_matrix().astype(np.float64)
    #train_x[0,6] = np.nan
    scored = gmm.score(train_x, 7, "egmm_model/shuttle.mdl", "egmm_model/score_out.csv")
    print metric(train_lbl, scored)

    import algorithm.pyad as pyft
    ff = pyft.IsolationForest()
    ff.train(train_x)
    score = ff.score(train_x)
    print metric(train_lbl, score)