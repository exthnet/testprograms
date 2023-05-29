#!/bin/bash -x
#PJM -L rscunit=ito-a
#PJM -L rscgrp=ito-ss
#PJM -L elapse=10:00
#PJM -L vnode-core=36
#PJM -j
#PJM -S

module load oneapi/2022.3.1
export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=/bin/pjrsh
export I_MPI_HYDRA_HOST_FILE=${PJM_O_NODEINF}
export I_MPI_FABRICS=shm:ofa
export I_MPI_PERHOST=4
export I_MPI_PIN_DOMAIN=4
export I_MPI_DEBUG=5
export OMP_NUM_THREADS=3
mpiexec.hydra -n 4 -print-rank-map numactl -l ./bcast_intelf 1000
mpiexec.hydra -n 4 -print-rank-map numactl -l ./bcast_intelf 1000
