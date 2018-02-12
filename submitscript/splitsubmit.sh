#!/bin/bash

# use current working directory for input and output
# default is to use the users home directory
#$ -cwd

# name this job
#$ -N motherset 

# send stdout and stderror to this file
#$ -o error/$TASK_ID.out
#$ -j y

# select queue - if needed 

# 5 cpu will be used. 
## $ -pe thread 5
# see where the job is being run
#hostname
hostname
#which python 
#export python=/
#print date and time
date
#set ff=$1
# Sleep for 20 seconds
#REP=$2
#BENCH_PATH=/nfs/guille/bugid/adams/meta_analysis/mothersets/
#BenchType=(binary multiclass regression);
BENCHMARK=$1
FIELD=$2
LABEL=$3
ALGORITHM=$6
OUTDIR=$5
MISSATT=$7
TYPE=$4
#echo $ALGORITHM
#for btype in "${BenchType[@]}"
#do
#if [ -f "$BENCH_PATH$btype/$BENCHMARK/${BENCHMARK}.preproc.csv" ]
#then
#DATASET="$BENCH_PATH$btype/$BENCHMARK/${BENCHMARK}.preproc.csv"
#fi 
#done
#Rscript motherset_trimm.R  $DATASET $BENCHMARK $SGE_TASK_ID $REP  
if [ `which python` == "/bin/python" ];
then
export PATH="/nfs/guille/bugid/adams/ifTadesse/anaconda2/bin:$PATH"
fi
if [ -z $MISSATT ];
then
    python splitjobs.py -i $BENCHMARK -c $FIELD -l $LABEL -n $SGE_TASK_ID -t $TYPE -o $OUTDIR -g $ALGORITHM
else
    python splitjobs.py -i $BENCHMARK -c $FIELD -l $LABEL -n $SGE_TASK_ID -t $TYPE -m $MISSATT -o $OUTDIR -g $ALGORITHM

fi
#print date and time again
date
