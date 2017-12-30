#!/bin/csh

# use current working directory for input and output
# default is to use the users home directory
#$ -cwd

# name this job
#$ -N genMod

# send stdout and stderror to this file
#$ -o debug/dbg_save_abalone_$TASK_ID.txt
#$ -j y

# select queue - if needed 
# -q eecs,eecs2,em64t,share

# see where the job is being run
hostname

# print date and time
date

#if ( ! -f models/model_abalone_$SGE_TASK_ID.mdl ) then
DATASET=$1
DIMS=$2
CLIST=$3
FNAME=/nfs/guille/bugid/adams/emmotta/bake_off/data/benchmarks/$DATASET/benchid_$DATASET$SGE_TASK_ID\_*.csv
/nfs/cluster-fserv/cluster-share/amran/testEGMM/osu_gmm_repository/osu-gmm/Release/gmm -file $FNAME -dims $DIMS -skipleftcols 2 -clusterlist $CLIST -ensemble 2 -replicates 15 -percentile 0.85 -ignoretiny -incremental -blocksize 15000 -savemodel -m models/model_$DATASET\_$SGE_TASK_ID.mdl -o output/out_$DATASET\_$SGE_TASK_ID.csv -debug
endif

# print date and time again
date

