#!/bin/bash
#SBATCH -p nova

hostname
date
. /etc/profile.d/modules.sh
module load compiler mpi

export I_MPI_DEBUG=5
export OMP_NUM_THREADS=2
export I_MPI_HYDRA_BOOTSTRAP=ssh

mpiexec.hydra -n 2 ./mpihello_hybrid_c

date

