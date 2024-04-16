#!/bin/bash
#SBATCH -o out_cublas_cu.txt
#SBATCH -p fire

module load nvhpc/24.1

sudo nvidia-smi -i 1 -pm 1
sudo nvidia-smi mig -cgi 1g.5gb,1g.5gb,1g.5gb,1g.5gb,1g.5gb,1g.5gb,1g.5gb -C
nvidia-smi
nvidia-smi -L
nvidia-smi -L | grep MIG | awk '{print $6}' | sed 's/)//' 2>&1 | tee mig.txt

./cublas_cu.out 4
# ./cublas_cu_2.out 8
