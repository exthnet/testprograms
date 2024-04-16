#!/bin/bash -x
#SBATCH -o out_cublas_acc_f.txt
#SBATCH -p fire

module load nvhpc/24.1

sudo nvidia-smi -i 1 -pm 1
sudo nvidia-smi mig -cgi 1g.5gb,1g.5gb,1g.5gb,1g.5gb,1g.5gb,1g.5gb,1g.5gb -C
nvidia-smi
nvidia-smi -L
nvidia-smi -L | grep MIG | awk '{print $6}' | sed 's/)//' 2>&1 | tee mig.txt

export NVCOMPILER_ACC_NOTIFY=3
export CUDA_VISIBLE_DEVICES=0
./cublas_acc_f.out 4
./cublas_acc_mf.out 4
