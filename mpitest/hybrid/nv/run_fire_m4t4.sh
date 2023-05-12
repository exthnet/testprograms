#!/bin/bash -x
#SBATCH -o out_fire_m4t4.txt
#SBATCH -p fire
#SBATCH -t 1:00

JOBID=`date "+%Y%m%d-%H%M%S"`

hostname
date
. /etc/profile.d/modules.sh
module load nvhpc/22.11

export OMP_NUM_THREADS=4
mpirun -n 4 -display-devel-map --map-by ppr:1:numa ./mpihello_hybrid_c
#mpirun -n 4 -display-devel-map ./mpihello_hybrid_c
