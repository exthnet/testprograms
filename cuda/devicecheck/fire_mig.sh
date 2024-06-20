#!/bin/bash -x
#SBATCH -p fire

module load nvhpc/24.3

sudo /usr/bin/systemctl stop nvidia-dcgm
sudo /usr/bin/nvidia-smi -mig 1
sudo /usr/bin/nvidia-smi -r
sudo /usr/bin/systemctl start nvidia-dcgm
sudo /usr/bin/nvidia-smi mig -cgi 1g.5gb,1g.5gb,1g.5gb,1g.5gb,1g.5gb,1g.5gb,1g.5gb -C
nvidia-smi
nvidia-smi -L
nvidia-smi -L | grep MIG | awk '{print $6}' | sed 's/)//' 2>&1 | tee mig.txt

mpirun -n 8 ./run2.sh ./check
