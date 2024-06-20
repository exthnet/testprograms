#!/bin/bash -x
#SBATCH -p fire

module load nvhpc/24.3

nvidia-smi
nvidia-smi -L

mpirun -n 2 ./check
