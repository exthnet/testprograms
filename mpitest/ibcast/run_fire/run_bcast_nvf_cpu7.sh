#!/bin/bash -x
#SBATCH -p fire

date
hostname
module load nvhpc/23.3
module list

export OMP_NUM_THREADS=1
mpirun -display-devel-map -n 7 --map-by ppr:2:numa ./bcast_nvf 800 2>&1|tee log_bcast_nvf_cpu7_800.txt

date
