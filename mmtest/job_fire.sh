#!/bin/bash
#SBATCH -p fire
hostname
date
. /etc/profile.d/modules.sh
#module load nvhpc/24.1
module load cuda/12.3.0

export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d
unset CUDA_VISIBLE_DEVICES

./out/mm_cuda -size 2000 -kernel 99
#nsys profile -o out_1 -f true --stats=true -t cuda,nvtx,osrt ./out/mm_cuda -size 4000 -kernel 99
nsys profile -o out_3a -f true --stats=true -t cuda,nvtx,osrt ./out/mm_cuda -size 2000 -kernel 99 -smlimit 15 &
nsys profile -o out_3b -f true --stats=true -t cuda,nvtx,osrt ./out/mm_cuda -size 2000 -kernel 99 -smlimit 15 &
nsys profile -o out_3c -f true --stats=true -t cuda,nvtx,osrt ./out/mm_cuda -size 2000 -kernel 99 -smlimit 15 &
wait


