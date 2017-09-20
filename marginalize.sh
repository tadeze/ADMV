#!/bin/sh

# use current working directory for input and output
# default is to use the users home directory
#$ -cwd

# name this job
#$ -N allDens

# send stdout and stderror to this file
#$ -o debug/dbg_load_$TASK_ID.txt
#$ -j y

# select queue - if needed
# -q eecs,eecs2,em64t,share

# see where the job is being run
hostname

# print date and time
date

DATASET=$1
DIMS=$2
CLIST=$3
FNAME=$4
#set FNAME=/nfs/guille/bugid/adams/emmotta/bake_off/data/benchmarks/$DATASET/benchid_$DATASET$SGE_TASK_ID\_*.csv
../osu_gmm_repository/osu-gmm/Release/gmm -file $DATASET -dims $DIMS -skipleftcols 1 -clusterlist $CLIST -ensemble 2 -replicates 15 -percentile 0.85 -ignoretiny -incremental -blocksize 15000 -loadmodel -m models/model_$FNAME$SGE_TASK_ID.mdl -explain -o ~/scratch/egmm/explanations/allDensity_$FNAME$SGE_TASK_ID.csv 

# print date and time again
date

