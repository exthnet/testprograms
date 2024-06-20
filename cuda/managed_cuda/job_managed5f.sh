#!/bin/bash -x
#SBATCH -p fire
date
hostname
. /etc/profile.d/modules.sh
module load nvhpc/24.5
nvidia-smi

TARGET=managed5f
export OMP_NUM_THREADS=4
nsys profile -o report_${TARGET}_${SLURM_JOBID}.qdrep --stats=true ./${TARGET}.out 1000 1000

date
