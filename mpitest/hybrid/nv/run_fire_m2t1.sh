#!/bin/bash -x
#SBATCH -p fire
#SBATCH -t 1:00

JOBID=`date "+%Y%m%d-%H%M%S"`

hostname
date
. /etc/profile.d/modules.sh
module load nvhpc/22.11

export OMP_NUM_THREADS=2
OPT="-display-devel-map"
mpirun -n 2 ${OPT} ./mpihello_hybrid_c
