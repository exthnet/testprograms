#!/bin/bash -x
#SBATCH -p fire
date
hostname
. /etc/profile.d/modules.sh
module load nvhpc/24.3
nvidia-smi
./gemm_managed3 2000
nsys profile -o report_${SLURM_JOBID}.qdrep --stats=true ./gemm_managed3 2000
date
