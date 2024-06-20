x:
	nvcc -O3 \
	-gencode arch=compute_60,code=\"sm_60,compute_60\" \
	-gencode arch=compute_61,code=\"sm_61,compute_61\" \
	-gencode arch=compute_70,code=\"sm_70,compute_70\" \
	-o shuffle shuffle.cu

shuffle1:
	nvcc -O3 \
	-gencode arch=compute_60,code=\"sm_60,compute_60\" \
	-gencode arch=compute_61,code=\"sm_61,compute_61\" \
	-gencode arch=compute_70,code=\"sm_70,compute_70\" \
	-o shuffle1 shuffle1.cu

all:
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	-arch=compute_60 -code=sm_70 \
	-arch=compute_60 -code=compute_70 \
	-o array_icc array.cu
	#-gencode arch=compute_70,code=sm_60 \
	#-gencode arch=compute_70,code=compute_60
#	nvcc -O3 -gencode arch=compute_60,code=sm_60 -o array array.cu
	nvcc -O3 \
	-arch=compute_60 -code=sm_70 \
	-arch=compute_60 -code=compute_70 \
	-o array array.cu

clean:
	-/bin/rm ./*~ ./*.o

