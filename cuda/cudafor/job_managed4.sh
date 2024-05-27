#!/bin/bash -x
#SBATCH -p fire-mig
date
hostname
. /etc/profile.d/modules.sh
module load nvhpc/24.3
nvidia-smi
./gemm_managed4 2000
nsys profile -o report_${SLURM_JOBID}.qdrep --stats=true ./gemm_managed4 2000
date
