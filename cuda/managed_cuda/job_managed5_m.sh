#!/bin/bash -x
#SBATCH -p fire-mig
date
hostname
. /etc/profile.d/modules.sh
module load nvhpc/24.5
nvidia-smi

TARGET=managed5
nsys profile -o report_${TARGET}m_${SLURM_JOBID}.qdrep --stats=true ./${TARGET}.out 1000 14

date
