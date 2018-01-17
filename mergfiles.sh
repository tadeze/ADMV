#!/bin/bash 
dirname=$1
for filename in $dirname/cell*.csv; do
tail -n +2 $filename >>cell_alldata.csv
echo "File .. $filename ..merged"
done
