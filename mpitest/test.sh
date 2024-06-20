#!/bin/bash
#SBATCH -p photon
. /etc/profile.d/modules.sh
module load nvhpc/23.1

mpirun -n 4 ./a.out
