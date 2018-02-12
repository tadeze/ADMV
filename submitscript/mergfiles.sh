#!/bin/bash 
dirname=$1
output=$2
for filename in $dirname/*.csv; do
tail -n +2 $filename >>$output
echo "File .. $filename ..merged"
done
