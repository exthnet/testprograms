#OPT=-lineinfo -gencode=arch=compute_80,code=sm_80
OPT=-lineinfo -arch=sm_80
all:
	nvcc ${OPT} -o default5e.out default5e.cu
	nvcc ${OPT} -o managed5e.out managed5e.cu
	nvcc ${OPT} -o pinned5e.out pinned5e.cu
	nvcc ${OPT} -Xcompiler -fopenmp -o default5f.out default5f.cu
	nvcc ${OPT} -o default6.out default6.cu
	nvcc ${OPT} -Xcompiler -fopenmp -o default6f.out default6f.cu
	nvcc ${OPT} -Xcompiler -fopenmp -o managed5f.out managed5f.cu
	nvcc ${OPT} -o managed5a.out managed5a.cu
	nvcc ${OPT} -o managed5b.out managed5b.cu
