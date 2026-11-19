#!/bin/bash

teststrings=(
"ROMEO" "JULIET"
#Test +*|
"ROMEO|JULIET" "R+" "R*" "R" "R+R*" "R*R+" "RR+" "RR*" "R+|J+" "(R|J)ULIET" "R*" 
#Test for the . wildcard
"R..EO" "R..EO|...IET" "R..*"
#Test for ranges
"[a-b]" "[q-s]" "[0-9]"
#Test for the \ sequence
"\?" "\|" "\+" "\*" "\." "\.\?\|\+\*" 
#Random tests
".*\?" "t*hi.\?"
);

prog=main

# no stale file ?
#rm -f RESULTS

date >> RESULTS
echo "" >> RESULTS
pass=1

len=${#teststrings[@]} 

FILES=../data/grep/testcases/*.txt

#loop over all test cases
for file in $FILES
do
	for ((i=0; i<len; i++))
	do
		testcase=${teststrings[$i]}
		if diff <(./$prog -f $file $testcase) <(egrep -x $testcase $file) >> RESULTS; then
			cat /dev/null
		else
			pass=0
			echo "Test Failed $testcase for $file"
		fi
	done
done

#special test for .*
for file in $FILES
do	
	if diff <(./$prog -f $file '.*') <(egrep -x '.*' $file) >> RESULTS; then
		cat /dev/null
	else
		pass=0
		echo "Test Failed .*"
	fi
done

if [ $pass -eq "1" ]; then
		echo "All tests passed" >> RESULTS
fi
	
echo "------------" >> RESULTS

