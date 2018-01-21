import os
from pandas import read_csv
import numpy as np
## Train
## - includes, train the model and save it.
## - score:
## - includes,, load the model & generate score from the model.
class Egmm:
    def __init__(self):
        self.gmm_path = "/home/tadeze/projects/ml/gmm/egmm"

    def train(self, file_name, dims, model_output, score_out, clusterlist="3,4,5",ensemble=2, replicates=15, percentile=0.85,
              block_size=15000, skip_cols=1):

        command = self.gmm_path+" -file {0:s} -dims {1:d} -clusterlist {2:s} -ensemble {3:d} -replicates {4:d} -percentile {5:f}" \
                  " -ignoretiny -incremental -blocksize {6:d} -savemodel -m {7:s} -o {8:s} -skipleftcols {9:d}".\
            format(file_name, dims, clusterlist, ensemble, replicates, percentile, block_size, model_output, score_out, skip_cols)

        #print command
        os.system(command)
    def score(self, test_x, dims, model_input, score_out):
        file_name = "egmm_model/texmp.txt"
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
        os.system(command)
        #print command

        nlscore = read_csv(score_out)["nlscore"].as_matrix()
        return nlscore

    #return read_csv(score_out)["score"].as_matrix()

if __name__ == '__main__':
    from util.common import metric
    file_name = "/home/tadeze/projects/missingvalue/datasets/anomaly/yeast/fullsamples/yeast_1.csv"
    gmm = Egmm()
    #gmm.train(file_name,7,"egmm_model/yeast.mdl","egmm_model/score.csv")
    score = gmm.score_file(file_name,7,"egmm_model/yeast.mdl", "egmm_model/score_out.csv")
    df = read_csv(file_name)
    train_lbl = map(int, df.ix[:, 0] == "anomaly")
    print metric(train_lbl, score)

    scored = gmm.score(df.ix[:,1:].as_matrix(), 7, "egmm_model/yeast.mdl", "egmm_model/score_out.csv")
    print metric(train_lbl, scored)
