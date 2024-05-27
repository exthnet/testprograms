#!/bin/bash -x
#SBATCH -p fire

date
. /etc/profile.d/modules.sh
module load nvhpc/24.1

export OMP_STACKSIZE=8m
ulimit -s unlimited

sudo nvidia-smi -i 1 -pm 1
sudo nvidia-smi mig -cgi 1g.5gb,1g.5gb,1g.5gb,1g.5gb,1g.5gb,1g.5gb,1g.5gb -C
nvidia-smi
nvidia-smi -L
nvidia-smi -L | grep MIG | awk '{print $6}' | sed 's/)//' 2>&1 | tee mig.txt

nsys profile -t nvtx,cuda,osrt,cublas --stats=true -o prof_${SLURM_JOBID} ./struct1 10 10 2


