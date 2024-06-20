#!/bin/sh

module purge
module load intel mpt
mpicc -qopenmp -O3 -o c.out ./hello.c
mpicc -qopenmp -O3 -o ct.out ./hello_t.c
mpif90 -qopenmp -O3 -o f.out ./hello.f
mpif90 -qopenmp -O3 -o ft.out ./hello_t.f
mpicc -qopenmp -O3 -mt_mpi -o cm.out ./hello.c
mpicc -qopenmp -O3 -mt_mpi -o cmt.out ./hello_t.c
mpif90 -qopenmp -O3 -mt_mpi -o fm.out ./hello.f
mpi90 -qopenmp -O3 -mt_mpi -o fmt.out ./hello_t.f
