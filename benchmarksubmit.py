import argparse
import os
import multiprocessing
from joblib import Parallel, delayed
def main():

    parser = argparse.ArgumentParser(description="Experiment parallel")
    #parser.add_argument('-i', '--input', type=argparse.FileType('r'),
     #                   help="Required input file", required=True)
    parser.add_argument('-c', '--column', help="Column hypheneted")
    parser.add_argument('-m', '--missing', help="Missing injection columns")
    parser.add_argument('-l', '--label', help="Ground flag label")
    parser.add_argument('-n', '--iteration', help="Number of iterations")
    parser.add_argument('-g', '--algorithm', help="Type of algorithm to use")
    #parser.add_argument('-n', '--iteration', help='Iteration size. Defualt 1')
    parser.add_argument('-t','--type', help="Experiment type.")
    parser.add_argument('-o', '--outputdir', help="Output directory location")

    args = parser.parse_args()
    return args

def submit_job(n):
    args = main()
    exp_type = "features"
    iteration = int(args.iteration)
    algorithm = ("loda", "ifor", "bifor", "lof")

    for algo in algorithm:
        command = "python mainexperiment.py -i " + args.input.name +" -c "+args.column+\
     " -l " + args.label+" -n "+str(n)+" -g "+ algo +" -t "+ args.type+\
     " -o "+args.outputdir
	os.system (command)
	return True
input_dir ="../dataset"
file_description = {'skin':3,
'magic.gamma':10, 'particle':50,'spambase':57, 'fault':27, 'gas':128,
'imgseg':18, 'landsat':36, 'letter.rec':16, 'opt.digits':61, 'pageb':10,'shuttle':9,
 'wave':21,'yeast':8, 'comm.and.crime':101,'abalone':7,'concrete':8,'wine':11, 'yearp':90,
                    'synthetic':10
}

def submit_benchmark_files():
    args = main()
    flag = int(args.label)
    for file_name in os.listdir(input_dir):
        bench_name = file_name.split("_")[0].split('.csv')[0]
        column = str(flag+1) + "-" + str(file_description[bench_name] + flag)
        full_path =  os.path.join(input_dir, file_name)
	if args.type=="del":
		command = "qdel {0:s}".format(bench_name)
		os.system(command)
		continue
	
	command = "qsub -t 1-"+args.iteration+ " -N "+bench_name+" submitscript/submitscript.sh "+ \
            full_path + " " + column+ " " + str(flag) + " "+ args.type + " " + args.outputdir +" " + \
        args.algorithm

        if bench_name=="particle": # or bench_name=="magic.gamma":
            continue
        os.system(command)
if __name__ == '__main__':
    submit_benchmark_files()
