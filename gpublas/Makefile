
LIBDIR=/home/share/nvidia/hpc_sdk/Linux_x86_64/24.1/math_libs/12.3/targets/x86_64-linux/lib

baseline: baseline.f90 baseline.c
	gfortran -fopenmp -o baseline_f.out baseline.f90
	gcc -fopenmp -o baseline_c.out baseline.c -lm
	nvfortran -fopenmp -o baseline_f_nv.out baseline.f90
	nvc -fopenmp -o baseline_c_nv.out baseline.c -lm

blas: blas.c blas.f90
	nvfortran -fopenmp -o blas_f_nv.out blas.f90 -lblas
	nvc -fopenmp -o blas_c_nv.out blas.c -lm -lblas

cublas: cublas_cf.f90 cublas_cu.cu
	nvfortran -cpp -cuda -lcublas -Minfo=accel -gpu=cc80 -cudalib=cublas,cusolver -g -o cublas_cf.out cublas_cf.f90
	nvcc -Xcompiler "-fopenmp" -arch=sm_80 -g -o cublas_cu.out cublas_cu.cu -lcublas
	# nvcc -Xcompiler "-fopenmp" -arch=sm_80 -g -o cublas_cu_2.out cublas_cu_2.cu -lcublas

cublas_acc: cublas_acc.c cublas_acc.f90 cublas_acc_m.c cublas_acc_m2.c cublas_acc_m.f90
	nvfortran -fopenmp -acc -lcublas -Minfo=accel -gpu=cc80 -cudalib=cublas -g -o cublas_acc_f.out cublas_acc.f90
	nvc -L${LIBDIR} -fopenmp -acc -lcublas -Minfo=accel -gpu=cc80 -g -o cublas_acc_c.out cublas_acc.c
	nvfortran -fopenmp -acc -lcublas -Minfo=accel -gpu=cc80,managed -cudalib=cublas -g -o cublas_acc_mf.out cublas_acc_m.f90
	#nvfortran -fopenmp -acc -lcublas -Minfo=accel -gpu=cc80,managed -cudalib=cublas -g -o cublas_acc_m2f.out cublas_acc_m2.f90
	nvc -L${LIBDIR} -fopenmp -acc -lcublas -Minfo=accel -gpu=cc80,managed -g -o cublas_acc_mc.out cublas_acc_m.c
	nvc -L${LIBDIR} -fopenmp -acc -lcublas -Minfo=accel -gpu=cc80,managed -g -o cublas_acc_m2c.out cublas_acc_m2.c

cublas_omp: cublas_omp.c cublas_omp.f90
	nvfortran -fopenmp -mp=gpu -lcublas -Minfo=accel -gpu=cc80 -cudalib=cublas -g -o cublas_omp_f.out cublas_omp.f90
	nvfortran -fopenmp -mp=gpu -lcublas -Minfo=accel -gpu=cc80 -cudalib=cublas -g -o cublas_omp_2f.out cublas_omp_2.f90
	nvc -L${LIBDIR} -fopenmp -mp=gpu -lcublas -Minfo=accel -gpu=cc80 -g -o cublas_omp_c.out cublas_omp.c
	nvfortran -fopenmp -mp=gpu -lcublas -Minfo=accel -gpu=cc80,managed -cudalib=cublas -g -o cublas_omp_mf.out cublas_omp_m.f90
	nvc -L${LIBDIR} -fopenmp -mp=gpu -lcublas -Minfo=accel -gpu=cc80,managed -g -o cublas_omp_mc.out cublas_omp_m.c

all: baseline cublas
clean:
	/bin/rm *~ *.out *.mod
