#!/bin/bash -x
#SBATCH -p fire
#SBATCH -t 1:00

JOBID=`date "+%Y%m%d-%H%M%S"`

hostname
date
. /etc/profile.d/modules.sh
module load nvhpc/22.11

numactl -H

export OMP_NUM_THREADS=4
OPT="-display-devel-map --map-by ppr:1:numa --bind-to numa"
mpirun -n 2 ${OPT} ./mpihello_hybrid_c
