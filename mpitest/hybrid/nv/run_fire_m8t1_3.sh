#!/bin/bash -x
#SBATCH -o out_fire_m8t1_3.txt
#SBATCH -p fire
#SBATCH -t 1:00

JOBID=`date "+%Y%m%d-%H%M%S"`

hostname
date
. /etc/profile.d/modules.sh
module load nvhpc/22.11

export OMP_NUM_THREADS=1
mpirun -n 8 -display-devel-map --bind-to core --map-by ppr:2:numa ./mpihello_hybrid_c
