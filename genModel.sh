#!/bin/sh

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
FNAME=$4
../osu_gmm_repository/osu-gmm/Release/gmm -file $DATASET -dims $DIMS -skipleftcols 1 -clusterlist $CLIST -ensemble 2 -replicates 15 -percentile 0.85 -ignoretiny -incremental -blocksize 15000 -savemodel -m models/model_$FNAME$SGE_TASK_ID.mdl -o ~/scratch/egmm/out_$FNAME$SGE_TASK_ID.csv -debug
# print date and time again
date

