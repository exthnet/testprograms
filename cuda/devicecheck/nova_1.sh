#!/bin/bash -x
#SBATCH -p nova

module load cuda/11.8.0

nvidia-smi
nvidia-smi -L

./check

export CUDA_VISIBLE_DEVICES="-1"
./check

export CUDA_VISIBLE_DEVICES="0"
./check

export CUDA_VISIBLE_DEVICES="1"
./check

export CUDA_VISIBLE_DEVICES="0,1"
./check

export CUDA_VISIBLE_DEVICES="1,0"
./check

export CUDA_VISIBLE_DEVICES="GPU-4ee1b2ff-50cf-5425-ecb3-f776f61b1600"
./check

