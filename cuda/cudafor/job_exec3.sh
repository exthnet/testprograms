#!/bin/bash -x
#SBATCH -p fire-mig
date
hostname
. /etc/profile.d/modules.sh
module load nvhpc/24.5
nvidia-smi
#./exec_managed3 10000
nsys profile -o report_${SLURM_JOBID}.qdrep --stats=false ./exec_managed3 10000
date
