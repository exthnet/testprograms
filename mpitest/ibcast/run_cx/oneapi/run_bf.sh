#!/bin/bash -x
#PJM -L rscgrp=cx-single
#PJM -L elapse=3:00
#PJM -j

date
hostname

module load oneapi
export OMP_NUM_THREADS=8
export I_MPI_DEBUG=10
export I_MPI_PIN_ORDER=compact
export I_MPI_PIN_DOMAIN=10
#export KMP_AFFINITY=verbose,granularity=fine,compact,1,0
#export I_MPI_HYDRA_BOOTSTRAP=ssh

n=1000
#mpiexec.hydra -n 4 -print-rank-map ./bcast_intelf ${n} #2>&1|tee log_bcast_intelf_${n}_omp${OMP_NUM_THREADS}.txt
mpiexec.hydra -n 4 -print-rank-map ./bcast_intelf ${n}
mpiexec.hydra -n 4 -print-rank-map ./bcast_intelf ${n}

date

