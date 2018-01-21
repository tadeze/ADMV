import argparse
import os
import multiprocessing
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description="Experiment parallel")
parser.add_argument('-i', '--input', type=argparse.FileType('r'),
                    help="Required input file", required=True)
parser.add_argument('-c', '--column', help="Column hypheneted")
parser.add_argument('-m', '--missing', help="Missing injection columns")
parser.add_argument('-l', '--label', help="Ground flag label")
parser.add_argument('-n', '--iteration', help="Number of iterations")
parser.add_argument('-g', '--algorithm', help="Type of algorithm to use")
#parser.add_argument('-n', '--iteration', help='Iteration size. Defualt 1')
parser.add_argument('-t','--type', help="Experiment type.")
parser.add_argument('-o', '--outputdir', help="Output directory location")

args = parser.parse_args()
exp_type="features"
iteration = int(args.iteration)
algorithm=("loda","ifor","bifor")
def submit_job(n):
	for algo in algorithm:
		command = "python mainexperiment.py -i " + args.input.name +" -c "+args.column+\
     " -l " + args.label+" -n "+str(n)+" -g "+ algo +" -t "+ args.type+\
     " -o "+args.outputdir
		os.system (command)
	return True


num_cores = multiprocessing.cpu_count()
result = Parallel(n_jobs=num_cores)(delayed(submit_job)(i) for i in range(5,iteration))
