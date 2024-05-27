#!/bin/bash -x
#SBATCH -p fire-mig
date
hostname
. /etc/profile.d/modules.sh
module load nvhpc/24.5
nvidia-smi
#./exec_managed4a 10000
nsys profile -o report_exec4a_${SLURM_JOBID}.qdrep --stats=true ./exec_managed4a 10000
date
