#!/bin/bash -x
#SBATCH -p fire

date
hostname
module load compiler mkl mpi
module list

export OMP_NUM_THREADS=4
export I_MPI_DEBUG=1
export I_MPI_PIN_ORDER=compact
export I_MPI_PIN_DOMAIN=4
export KMP_AFFINITY=verbose,granularity=fine,compact,1,0
export I_MPI_HYDRA_BOOTSTRAP=ssh
export I_MPI_ASYNC_PROGRESS=1

mpiexec.hydra -n 4 ./ibcast_intel 1000

date
