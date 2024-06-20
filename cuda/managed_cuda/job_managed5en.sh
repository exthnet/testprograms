#!/bin/bash -x
#SBATCH -p fire
date
hostname
. /etc/profile.d/modules.sh
module load nvhpc/24.5
nvidia-smi

TARGET=managed5e
nsys profile -o report_${TARGET}n_${SLURM_JOBID}.qdrep --stats=true ./${TARGET}.out 10000

date
