#!/bin/bash -x
#SBATCH -p fire-mig
date
hostname
. /etc/profile.d/modules.sh
module load nvhpc/24.5
nvidia-smi

nvidia-smi
# ./managed.out 2000
nsys profile -o report_managed_${SLURM_JOBID}.qdrep --stats=true ./managed.out 2000

date
