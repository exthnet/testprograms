#!/bin/bash

function A(){
	cmd="paste -d,"
	for n in 32 16 8 4 2 1
	do
		if [ -e tmp_A${n}.txt ]; then rm tmp_A${n}.txt; fi
		for m in `seq 1 32`
		do
#			echo "A${m}MW${n}r"
			grep A${m}MW${n}r $1 | tail -n 1 | awk '{print $3}' >> tmp_A${n}.txt
		done
		cmd="${cmd} tmp_A${n}.txt"
	done
	echo ${cmd}
}

function A(){
	cmd="paste -d,"
	for n in 32 16 8 4 2 1
	do
		if [ -e tmp_B${n}.txt ]; then rm tmp_B${n}.txt; fi
		for m in `seq 1 32`
		do
#			echo "B${m}MW${n}r"
			grep B${m}MW${n}r $1 | tail -n 1 | awk '{print $3}' >> tmp_B${n}.txt
		done
		cmd="${cmd} tmp_B${n}.txt"
	done
	echo ${cmd}
}

A log_globalA2.txt
B log_globalA2.txt
