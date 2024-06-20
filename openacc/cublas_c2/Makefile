# -*- Makefile -*-

TARGETS=cpu cublas acc hybrid
all: ${TARGETS}

cpu: cpu.c
	pgcc -O3 -o cpu cpu.c

cublas: cublas.c
	nvcc -O3 -lcublas -o cublas cublas.c

acc: acc.c
	pgcc -Mcuda=cc20 -acc -O3 -Minfo -ta=tesla,cc20 -o acc acc.c

hybrid: hybrid.c
	pgcc -Mcuda=cc20 -acc -O3 -Minfo -ta=tesla,cc20 -lcublas -o hybrid hybrid.c

#cuda: cuda.cu
#	nvcc -O3 -o cuda cuda.cu

clean:
	-/bin/rm ./*~ ./*.o
	-/bin/rm ${TARGETS}
