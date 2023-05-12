#!/bin/bash -x
#SBATCH -o out_fire_m16t1_1.txt
#SBATCH -p fire
#SBATCH -t 1:00

JOBID=`date "+%Y%m%d-%H%M%S"`

hostname
date
. /etc/profile.d/modules.sh
module load nvhpc/22.11

export OMP_NUM_THREADS=1
mpirun -n 16 -display-devel-map --bind-to core ./mpihello_hybrid_c
