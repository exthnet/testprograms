#!/bin/sh
#------pjsub option ------#
#PJM -L "rscgrp=short"
#PJM -L "node=1"
#PJM -L "elapse=10:00"
#PJM -g gc26
#------Proguram exection ------#

date
cd ${PJM_O_WORKDIR}
export OMP_NUM_THREADS=16

run () {
	../out/mm_cbcg -m $1 -n $2 -k $3 -loop 10 -kernel $4
}

for i in 0 1 2 3 55
do
	run 1000 1000 1000 $i
done

# exit

for i in 0 1 2 3 55
do
	run 1000000 10 10 $i
done

for i in 0 1 2 3 55
do
	run 10 1000000 10 $i
done

for i in 0 1 2 3 55
do
	run 10 10 1000000 $i
done


for i in 0 1 2 3 55
do
	run 1000000 16 16 $i
done

for i in 0 1 2 3 55
do
	run 16 1000000 16 $i
done

for i in 0 1 2 3 55
do
	run 16 16 1000000 $i
done


for i in 0 1 2 3 55
do
	run 1000000 32 32 $i
done

for i in 0 1 2 3 55
do
	run 32 1000000 32 $i
done

for i in 0 1 2 3 55
do
	run 32 32 1000000 $i
done
