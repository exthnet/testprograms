#!/bin/bash -x
#SBATCH -p fire

date
hostname
module load nvhpc/23.3
module list

sudo nvidia-smi -i 1 -pm 1
sudo nvidia-smi mig -cgi 1g.5gb,1g.5gb,1g.5gb,1g.5gb,1g.5gb,1g.5gb,1g.5gb -C
nvidia-smi
nvidia-smi -L

export OMP_NUM_THREADS=1
mpirun -display-devel-map -n 7 ./run7.sh ./bcast_managed_nvf 100 2>&1|tee log_bcast_managed_nvf_mig7_100.txt

date
