all:
	#nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" -gencode arch=compute_60,code=sm_60 -o array_icc array.cu
	#nvcc -O3 -gencode arch=compute_60,code=sm_60 -o array array.cu
	gcc -o cpu cpu.c
	nvcc -O3 -o cuda cuda.cu
	nvcc -O3 -o managed managed.cu
	#nvcc -O3 -gencode arch=compute_60,code=sm_60 -o managed managed.cu

clean:
	-/bin/rm ./*~ ./*.o
