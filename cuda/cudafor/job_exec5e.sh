#!/bin/bash -x
#SBATCH -p fire-mig
date
hostname
. /etc/profile.d/modules.sh
module load nvhpc/24.5
nvidia-smi
#./exec_managed5e 10000
nsys profile -o report_exec5e_${SLURM_JOBID}.qdrep --stats=true ./exec_managed5e 10000
date
