if:
	mpiifort -O3 -qopenmp -o bcast_intelf bcast.f90
	mpiifort -O3 -qopenmp -o ibcast_intelf ibcast.f90
nvf:
	mpifort -O3 -mp -o bcast_nvf bcast.f90
	mpifort -O3 -mp -o ibcast_nvf ibcast.f90
ic:
	mpiicc -O3 -qopenmp -o bcast_intelc bcast.c
	mpiicc -O3 -qopenmp -o ibcast_intelc ibcast.c
nvc:
	mpicc -O3 -mp -o bcast_nvc bcast.c
	mpicc -O3 -mp -o ibcast_nvc ibcast.c
