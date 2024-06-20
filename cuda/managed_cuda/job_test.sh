#!/bin/bash -x
#SBATCH -p fire-mig
date
hostname
. /etc/profile.d/modules.sh
module load nvhpc/24.5
nvidia-smi

nvidia-smi
nsys profile -o report_test_${SLURM_JOBID}.qdrep --stats=true ./test.out

date
