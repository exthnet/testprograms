# -*- Makefile -*-

TARGETS=cuda_c hybrid1 hybrid2
all: ${TARGETS}

hybrid1: acc_main.o cudakernel.o
	pgcc -Mcuda=cc20 -acc -o hybrid1 cudakernel.o acc_main.o
acc_main.o: acc_main.c
	pgcc -acc -O3 -Minfo -ta=tesla,cc20 -c acc_main.c
cudakernel.o: cudakernel.cu
	nvcc -O3 -gencode arch=compute_20,code=\"sm_20,compute_20\" -c cudakernel.cu

cuda_c: cuda_c.cu
	nvcc -O3 -gencode arch=compute_20,code=\"sm_20,compute_20\" -o cuda_c cuda_c.cu

hybrid2: cuda_main.o acckernel.o
	pgcc -Mcuda=cc20 -acc -o hybrid2 acckernel.o cuda_main.o
cuda_main.o: cuda_main.cu
	nvcc -O3 -gencode arch=compute_20,code=\"sm_20,compute_20\" -c cuda_main.cu
acckernel.o: acckernel.c
	pgcc -acc -O3 -Minfo -ta=tesla,cc20 -c acckernel.c

clean:
	-/bin/rm ./*~ ./*.o
	-/bin/rm ${TARGETS}
