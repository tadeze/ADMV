import argparse
import os
import multiprocessing
#from joblib import Parallel, delayed


def main():
    parser = argparse.ArgumentParser(description="Experiment parallel")
    parser.add_argument('-i', '--input', type=argparse.FileType('r'),
                       help="Required input file", required=False)
    parser.add_argument('-c', '--column', help="Column hypheneted")
    parser.add_argument('-m', '--missing', help="Missing injection columns")
    parser.add_argument('-l', '--label', help="Ground flag label")
    parser.add_argument('-n', '--iteration', help="Number of iterations")
    parser.add_argument('-g', '--algorithm', help="Type of algorithm to use")
    #parser.add_argument('-n', '--iteration', help='Iteration size. Defualt 1')
    parser.add_argument('-t', '--type', help="Experiment type.")
    parser.add_argument('-o', '--outputdir', help="Output directory location")
    parser.add_argument('-s','--server', help="Server type. Either `cluter` or `local`")
    parser.add_argument('-b', '--bench', help="Benchmark nam")
    parser.add_argument('-d', '--inputdir', help="Input directory.")

    args = parser.parse_args()
    return args


def submit_job(n):
    args = main()
    exp_type = "features"
    iteration = int(args.iteration)
    algorithm = ("loda", "ifor", "bifor", "lof")

    for algo in algorithm:
        command = "python mainexperiment.py -i " + args.input.name + " -c " + args.column + \
                  " -l " + args.label + " -n " + str(n) + " -g " + algo + " -t " + args.type + \
                  " -o " + args.outputdir
        os.system(command)
        return True


#input_dir = "../group2"
input_dir = "synthetic"
file_description = {'skin': 3,
                    'magic.gamma': 10, 'particle': 50, 'spambase': 57, 'fault': 27, 'gas': 128,
                    'imgseg': 18, 'landsat': 36, 'letter.rec': 16, 'opt.digits': 61, 'pageb': 10, 'shuttle': 9,
                    'wave': 21, 'yeast': 8, 'comm.and.crime': 101, 'abalone': 7, 'concrete': 8, 'wine': 11, 'yearp': 90,
                    'synthetic': 8
                    }

def parallel_local_single_batch(file_name, n):
    bench_name = os.path.basename(file_name).split("_")[0].split('.csv')[0]
    flag = int(args.label)
    column = str(flag + 1) + "-" + str(file_description[bench_name] + flag)
    full_path = os.path.join(input_dir, file_name)
    if donot_run_these(bench_name):
        return 0

    command = "python splitjobs.py -i {0:s} -c {1:s} -l {2:s} -n {3:d} -g {4:s} -t {5:s} -o {6:s}".\
    format(full_path, column, args.label, n, args.algorithm, args.type, args.outputdir)
    #print command
    os.system(command)
    return 1
from joblib import Parallel, delayed
def run_split_paralell():
    all_files = os.listdir(input_dir)
    num_cores = multiprocessing.cpu_count()
    #print num_cores
    #pool = multiprocessing.Pool(num_cores)
    #pool.map(parallel_local_single_batch, range(1,10))
    #
    for file_name in all_files:
        bench_name = os.path.basename(file_name).split("_")[0].split('.csv')[0]
        if bench_name==args.bench:

            num_cores = multiprocessing.cpu_count()
            result = Parallel(n_jobs=num_cores)(delayed(parallel_local_single_batch)(file_name, i) for i in range(1, 89))



def parallel_local(file_name):

    bench_name = file_name.split("_")[0].split('.csv')[0]
    flag = int(args.label)
    column = str(flag + 1) + "-" + str(file_description[bench_name] + flag)
    full_path = os.path.join(input_dir, file_name)
    if donot_run_these(bench_name):
        return 0
    command = "python mainexperiment.py -i {0:s} -c {1:s} -l {2:s} -n {3:s} -g {4:s} -t {5:s} -o {6:s}".\
    format(full_path, column, args.label, args.iteration, args.algorithm, args.type, args.outputdir)
    os.system(command)
    return 1

def run_parallel():

    all_files = os.listdir(input_dir)

    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    pool.map(parallel_local, all_files)
    #result = Parallel(n_jobs=num_cores)(delayed(parallel_local)(file_name) for file_name in all_files)
def donot_run_these(bench_name):
    if bench_name in ["particle", "gas", "yeast", "opt.digits","comm.and.crime",
                      "yearp"]:# or bench_name=="magic.gamma":
        return True


def submit_benchmark_files():
    #args = main()
    flag = int(args.label)
    for file_name in os.listdir(input_dir):
        bench_name = file_name.split("_")[0].split('.csv')[0]
        column = str(flag + 1) + "-" + str(file_description[bench_name] + flag)
        full_path = os.path.join(input_dir, file_name)

        if args.type == "del":
            command = "qdel {0:s}_{1:s}".format(args.algorithm, bench_name)
            os.system(command)
            continue
        t_name = int(file_name.split("_")[2].split(".csv")[0])
        command = "qsub -t " + str(t_name) + " -N " + args.algorithm + "_" + bench_name + " submitscript/submitscript.sh " + \
                  full_path + " " + column + " " + str(flag) + " " + args.type + " " + args.outputdir + " " + \
                  args.algorithm
        if donot_run_these(bench_name):
            continue

        #print command

        os.system(command)

def split_submit_benchmark_files():
    #args = main()
    output_dir = "/scratch/cluster-share/zemicheal/missingdata/kddexp/benchmark1/"
    algorithms = "LODA,IFOR,BIFOR"
    flag = int(args.label)
    if args.outputdir is not None:
        output_dir = args.outputdir
    if args.algorithm is not None:
        algorithms = args.algorithm
    if args.inputdir is not None:
        input_dir = args.inputdir
    for file_name in os.listdir(input_dir):
        fsplit = file_name.split("_")
        bench_name = fsplit[0].split('.csv')[0]
        file_id = fsplit[1]
       # print file_id
        column = str(flag + 1) + "-" + str(file_description[bench_name] + flag)
        full_path = os.path.join(input_dir, file_name)

        if args.type == "del":
            command = "qdel {0:s}".format(file_name)
            os.system(command)
            continue
        t_name = 90 #int(file_name.split("_")[2].split(".csv")[0])
        #if bench_name =="spambase":

        command = "qsub -N " + bench_name+"_"+str(t_name)+ " -t 1-99 submitscript/splitsubmit.sh " + \
                  full_path + " " + column + " " + str(flag) + " splitjobs " + output_dir + " " + \
                  algorithms
        if donot_run_these(bench_name):
            continue
        if args.bench is not None:
            if bench_name == args.bench:
                os.system(command)
        else:
            #print command
            if bench_name in ["fault","pageb","spambase"]:
                os.system(command)
            #break
from pandas import DataFrame,read_csv, Series
from totalcorrelation import total_correlation
import numpy as np
def total_information():
    flag = int(args.label)
    total_corr = DataFrame()
    if args.outputdir is not None:
        output_dir = args.outputdir
    if args.inputdir is not None:
        input_dir = args.inputdir
    for file_name in os.listdir(input_dir):
        fsplit = file_name.split("_")
        bench_name = fsplit[0].split('.csv')[0]
        file_id = fsplit[1]
        # print file_id
        column = str(flag + 1) + "-" + str(file_description[bench_name] + flag)
        full_path = os.path.join(input_dir, file_name)
        if donot_run_these(bench_name):
            continue

        if args.type == "totalcorrelation":
            df = read_csv(full_path)
            
	    train_x = df[df['class']=="nominal"].ix[:,(flag+1):(flag+file_description[bench_name])]
            train_x = train_x.as_matrix().astype(np.float64)
            current_total_corr = Series([bench_name, file_name, total_correlation(train_x)])
            total_corr = total_corr.append(current_total_corr,ignore_index=True)
    total_corr.rename(columns={0:'benchmark',1:'filename',2:'totalcorrelation'},inplace=True)
    total_corr.to_csv("total_correlation.csv",na_rep=np.nan)

if __name__ == '__main__':
    args = main()
    if args.server == "cluster":
        submit_benchmark_files()
    elif args.server == "local":
            run_split_paralell()
    elif args.server =="split":
        split_submit_benchmark_files()
    elif args.server == "entropy":
        total_information()
    else:
        ValueError("Invalid experiment type")
