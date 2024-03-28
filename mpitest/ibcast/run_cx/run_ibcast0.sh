#!/bin/bash -x
#PJM -L rscgrp=cx-single
#PJM -L elapse=3:00
#PJM -j

date
hostname

module load oneapi
export OMP_NUM_THREADS=4
export I_MPI_DEBUG=10
export I_MPI_PIN_ORDER=compact
export I_MPI_PIN_DOMAIN=10
#export KMP_AFFINITY=verbose,granularity=fine,compact,1,0
#export I_MPI_HYDRA_BOOTSTRAP=ssh

env

mpiexec.hydra -n 4 ./run2.sh ./ibcast_intel 10

date
