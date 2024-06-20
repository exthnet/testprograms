#!/bin/sh

module purge
module load intel intel-mpi
mpiicc -qopenmp -O3 -o c.out ./hello.c
mpiicc -qopenmp -O3 -o ct.out ./hello_t.c
mpiifort -qopenmp -O3 -o f.out ./hello.f
mpiifort -qopenmp -O3 -o ft.out ./hellot.f
mpiicc -qopenmp -O3 -mt_mpi -o cm.out ./hello.c
mpiicc -qopenmp -O3 -mt_mpi -o cmt.out ./hello_t.c
mpiifort -qopenmp -O3 -mt_mpi -o fm.out ./hello.f
mpiifort -qopenmp -O3 -mt_mpi -o fmt.out ./hellot.f

