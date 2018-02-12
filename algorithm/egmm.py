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
        file_name = os.path.join(tempresult, os.path.basename(model_input).split('.csv')[0]+str(num)+"_texmp.txt")
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
        os.system(command)

        nlscore = read_csv(score_out)["nlscore"].as_matrix()
        return nlscore
