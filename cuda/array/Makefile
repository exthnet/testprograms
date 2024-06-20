all:
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" -gencode arch=compute_60,code=sm_60 -o array_icc array.cu
	nvcc -O3 -gencode arch=compute_60,code=sm_60 -o array array.cu

clean:
	-/bin/rm ./*~ ./*.o
