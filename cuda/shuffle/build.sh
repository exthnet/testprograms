#!/bin/bash

for f in shuffle25 shuffle26 shuffle27 shuffle28 shuffle29 shuffle30 #shuffle10 #shuffle2 shuffle3 shuffle4 shuffle5 shuffle6 shuffle7 shuffle8 shuffle9
do
	nvcc -O3 -arch=compute_60 -code=sm_60 -o ${f}_66 ${f}.cu
	nvcc -O3 -arch=compute_60 -code=sm_70 -o ${f}_67 ${f}.cu
	nvcc -O3 -arch=compute_60 -code=sm_60,sm_70 -o ${f}_667 ${f}.cu
	nvcc -O3 -arch=compute_70 -code=sm_70 -o ${f}_77 ${f}.cu
done

