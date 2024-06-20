#!/bin/sh
#PBS -q u-debug
#PBS -l select=1:mpiprocs=2
#PBS -W group_list=gc26f
# -W group_list=jh160041
#PBS -l walltime=10:00

cd ${PBS_O_WORKDIR}
env
. /etc/profile.d/modules.sh
module load intel intel-mpi
ulimit -s unlimited
export OMP_NUM_THREADS=2

mpirun -n 2 ./c.out
mpirun -n 2 ./ct.out
mpirun -n 2 ./f.out
mpirun -n 2 ./ft.out
mpirun -n 2 ./cm.out
mpirun -n 2 ./cmt.out
mpirun -n 2 ./fm.out
mpirun -n 2 ./fmt.out
