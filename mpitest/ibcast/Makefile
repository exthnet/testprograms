gf:
	mpifort -O3 -fopenmp -o bcast_gf bcast.f90
	mpifort -O3 -fopenmp -o ibcast_gf ibcast.f90
if:
	mpiifort -O3 -qopenmp -o bcast_intelf bcast.f90
	mpiifort -O3 -qopenmp -o ibcast_intelf ibcast.f90
nvf:
	mpifort -O3 -mp -o bcast_nvf bcast.f90
	mpifort -O3 -mp -o ibcast_nvf ibcast.f90

gc:
	mpicc -O3 -fopenmp -o bcast_gc bcast.c
	mpicc -O3 -fopenmp -o ibcast_gc ibcast.c
ic:
	mpiicc -O3 -qopenmp -o bcast_intelc bcast.c
	mpiicc -O3 -qopenmp -o ibcast_intelc ibcast.c
nvc:
	mpicc -O3 -mp -o bcast_nvc bcast.c
	mpicc -O3 -mp -o ibcast_nvc ibcast.c

nvf2:
	mpifort -O3 -tp=zen3 -cuda -cudalib=cublas,cusolver -mp -o bcast_dev_nvf bcast_dev.f90
	mpifort -O3 -tp=zen3 -cuda -cudalib=cublas,cusolver -mp -o bcast_managed_nvf bcast_managed.f90
