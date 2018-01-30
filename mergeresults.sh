#!/bin/bash 
# Merge files located at dirname and put in outputpath
# mergeresult <dirname> <outiput>

dirname=$1
outputpath=$2
for algo in $dirname/*; do
	echo $algo
	for benchmark in $algo/*;
	do
		tail -n +2 $benchmark >> $outputpath
		echo "File ...$benchmark ... merged"
	#   	output=$outputpath/`basename $algo`_`basename $benchmark`.csv
		
	#	for filename in $benchmark/*.csv; do
	#		tail -n +2 $filename >>$output
	#		echo "File .. $filename ..merged"
	#		echo $benchmark.`basename $filename`
	#	done
		#echo $output
		#echo "\n"
	done
done
