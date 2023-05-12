#!/bin/bash -x
#PJM -L rscgrp=cx-single
#PJM -L elapse=3:00
#PJM -j

date
hostname

module load hpc_sdk/23.1
export OMP_NUM_THREADS=4

env

n=800
mpirun -display-devel-map -n 4 --map-by ppr:2:socket ./bcast_nvf ${n} 2>&1|tee log_bcast_nvf_${n}.txt

date
