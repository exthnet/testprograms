#!/bin/bash -x
#SBATCH -p fire

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

