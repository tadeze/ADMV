#FIELD=5
#LABEL=4
ALGORITHM=('loda' 'ifor' 'bifor')
BENCHMARK=$1
FIELD=$2
LABEL=$3
TYPE=$4
NUMITER=$5
MISSATT=$7
OUTPUTDIR=$6
# if [ `which python` == "/bin/python" ];
# then
# export PATH="/nfs/guille/bugid/adams/ifTadesse/anaconda2/bin:$PATH"
# fi
#which python
for N in $(seq 1 $NUMITER)
do
	for ALGO in "${ALGORITHM[@]}"
	do 
		if [ -z $MISSATT ];
		then
			python mainexperiment.py -i $BENCHMARK -c $FIELD -l $LABEL -n $N -g $ALGO -t $TYPE -o $OUTPUTDIR
		else
			python mainexperiment.py -i $BENCHMARK -c $FIELD -l $LABEL -n $N -g $ALGO -t $TYPE -m $MISSATT

		fi
	echo "$ALGO  on iteration $N completed ..."
	done
	echo "Iteration $N completed"
done
#name=`basename $benchmark`
# EXP_TYPE=$1
# #Dataset location
# BENCH_PATH=/nfs/guille/bugid/adams/meta_analysis/mothersets/
# BenchType=(binary multiclass regression);
# #echo $ALGORITHM
# for btype in "${BenchType[@]}"
# do
# 	datasets=`ls $BENCH_PATH$btype`
# 	#echo datasets
# 	for BENCHMARK in $BENCH_PATH$btype/*
# 	do
# 		BENCHNAME=`basename $BENCHMARK`
# 		#echo "$BENCHMARK/${benchname}.preproc.csv"

# 		DATASET="$BENCHMARK/${BENCHNAME}.preproc.csv"
# 		#echo $DATASET
# 		if [ -f $DATASET ]
# 		then
# 			for ALGO in "${ALGORITHM[@]}"
# 			do

#  				qsub -N $BENCHNAME -t 1-30 submitscript/submitscript.sh $DATASET $FIELD $LABEL $ALGO $EXP_TYPE
#  			done

# #			echo "$DATASET"
# 		fi
# 	done

# done


# for alg in "${algs[@]}"
# 	do
# 		qsub -N $name -t 1-30 submitscript/submitscript.sh $benchmark $field $label $alg $etype
# done



