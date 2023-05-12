intel:
	mpiifort -O3 -qopenmp -o bcast_intel bcast.f90
	mpiifort -O3 -qopenmp -o ibcast_intel ibcast.f90
nv:
	mpifort -O3 -mp -o bcast_nv bcast.f90
	mpifort -O3 -mp -o ibcast_nv ibcast.f90
ic:
	mpiicc -O3 -qopenmp -o bcast_intelc bcast.c
	mpiicc -O3 -qopenmp -o ibcast_intelc ibcast.c
nvc:
