#
META=$1

for i in "a" "b" "c"
do
	echo $i
done


if [ -z $META ];
then 
	echo "No argument given"
else
	echo $META
fi
