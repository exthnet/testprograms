#!/bin/bash -x
#SBATCH -p fire

date
hostname
module load nvhpc/23.3
module list

sudo nvidia-smi -i 1 -pm 1
sudo nvidia-smi mig -cgi 7g.40gb -C
nvidia-smi
nvidia-smi -L

export CUDA_VISIBLE_DEVICES=MIG-33f7f727-ed51-5ce0-96cf-952be15d47c4
export OMP_NUM_THREADS=1
mpirun -display-devel-map -n 7 --map-by ppr:2:numa --mca btl_smcuda_use_cuda_ipc 0 ./bcast_dev_nvf 100 2>&1|tee log_bcast_dev_nvf_gpu7_100.txt

date
