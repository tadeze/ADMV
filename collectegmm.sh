#SCORE=~/scratch/egmm/*.csv
#EXP=~scratch/egmm/explanations/*.csv
DATASET=$1

for i in {1..10}
do
SCORE=~/scratch/egmm/out_$DATASET$i.csv
EXP=~/scratch/egmm/explanations/allDensity_$DATASET$i.csv
RAW=~/adams/missingdata/experiments/dataset/$DATASET.csv
python egmm_miss.py -e $EXP -i $RAW -s $SCORE -n $i
#echo $SCORE
#echo $EXP
#echo $RAW
done
