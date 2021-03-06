#!/usr/bin/env bash
#benchmark=$1
#field=$2
#label=$3
##
## First argument is experiment type. 
###   {cell, feature}
FIELD=5
LABEL=4
ALGORITHM=('loda' 'ifor') # 'bifor')
#name=`basename $benchmark`
EXP_TYPE=$1
OUTDIR=/scratch/cluster-share/zemicheal/missingdata/kddexp/motherset
#Dataset location 
BENCH_PATH=/nfs/guille/bugid/adams/meta_analysis/mothersets
BenchType=(binary multiclass regression);
#echo $ALGORITHM
for btype in "${BenchType[@]}"
do
	datasets=`ls $BENCH_PATH$btype`
	#echo datasets
	for BENCHMARK in $BENCH_PATH$btype/*
	do
		BENCHNAME=`basename $BENCHMARK`
		#echo "$BENCHMARK/${benchname}.preproc.csv"
	
		DATASET="$BENCHMARK/${BENCHNAME}.preproc.csv"
		#echo $DATASET 
		if [ -f $DATASET ]
		then
			for ALGO in "${ALGORITHM[@]}"
			do

 				qsub -N $BENCHNAME -t 1-30 submitscript/submitscript.sh $DATASET $FIELD $LABEL $ALGO $EXP_TYPE $OUTDIR
 			done	

#			echo "$DATASET"
		fi
	done

done


