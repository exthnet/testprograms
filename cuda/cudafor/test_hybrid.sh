#!/bin/bash -x
#PBS -l select=1:slot_type=NC24ads_A100_v4
#PBS -j oe

cd ${PBS_O_WORKDIR}

hostname
date
. /etc/profile.d/modules.sh
module use /mnt/sw/modulefiles
#module load nvhpc/22.11
#module load nvhpc/23.3
module load gcc-9.2.0
module load nvidia/nvhpc/23.3

export OMP_NUM_THREADS=4
#mpirun -display-devel-map -n 2 ./hybrid

nvidia-smi
nvidia-smi -L

mpirun -display-devel-map -n 2 ./hybrid_cc80
mpirun -display-devel-map -n 2 ./hybrid_ccall

date
