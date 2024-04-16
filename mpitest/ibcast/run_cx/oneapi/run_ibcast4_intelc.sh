#!/bin/bash -x
#PJM -L rscgrp=cx-single
#PJM -L elapse=3:00
#PJM -j

date
hostname

module load oneapi
export OMP_NUM_THREADS=1
export I_MPI_DEBUG=10
export I_MPI_PIN_ORDER=compact
export I_MPI_PIN_DOMAIN=4
#export KMP_AFFINITY=verbose,granularity=fine,compact,1,0
#export I_MPI_HYDRA_BOOTSTRAP=ssh
export I_MPI_ASYNC_PROGRESS=1
export I_MPI_ASYNC_PROGRESS_THREADS=1

n=5000
mpiexec.hydra -n 4 -print-rank-map numactl -l ./ibcast_intelc ${n}
mpiexec.hydra -n 4 -print-rank-map numactl -l ./ibcast_intelc ${n}
mpiexec.hydra -n 4 -print-rank-map numactl -l ./ibcast_intelc ${n}

date
