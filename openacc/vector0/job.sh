#!/bin/bash
#SBATCH -p photon
#SBATCH -o out.txt
#SBATCH -t 1:00:00
date
hostname
. /etc/profile.d/modules.sh
module load nvhpc/24.3
nvidia-smi
./vector_c
date
