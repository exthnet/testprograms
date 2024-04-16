#!/bin/bash -x
#PJM -L rscgrp=cx-share
#PJM -L elapse=3:00
#PJM -j

date
hostname

module load oneapi
export OMP_NUM_THREADS=4
export I_MPI_DEBUG=1
export I_MPI_PIN_ORDER=compact
export I_MPI_PIN_DOMAIN=4
#export KMP_AFFINITY=verbose,granularity=fine,compact,1,0
#export I_MPI_HYDRA_BOOTSTRAP=ssh

mpiexec.hydra -n 4 ./ibcast_intel 100

date
