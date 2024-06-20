pgi:
	pgfortran -O2 -Minfo laplace2.f90 -o laplace2-F
	pgfortran -O2 -acc -Minfo=accel laplace2.f90 -o laplace2acc-F
gnu:
	gfortran -fopenmp -O3 laplace2.f90 -o laplace2-F_gnu
	gfortran -fopenacc -foffload=nvptx-none -offload="-O3" -O3 laplace2.f90 -o laplace2acc-F_gnu

