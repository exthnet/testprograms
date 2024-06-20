coalesced: coalesced.cu
	nvcc -O3 -Xcompiler="-fopenmp" \
	-gencode arch=compute_60,code=\"sm_60,compute_60\" \
	-gencode arch=compute_61,code=\"sm_61,compute_61\" \
	-gencode arch=compute_70,code=\"sm_70,compute_70\" \
	-o coalesced coalesced.cu

i:
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	-gencode arch=compute_60,code=\"sm_60,compute_60\" \
	-gencode arch=compute_61,code=\"sm_61,compute_61\" \
	-gencode arch=compute_70,code=\"sm_70,compute_70\" \
	-o coalesced coalesced.cu


clean:
	-/bin/rm ./*~ ./*.o

