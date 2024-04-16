#!/bin/bash -x
#PJM -L rscgrp=cx-single
#PJM -L elapse=3:00
#PJM -j

date
hostname

module load hpc_sdk/23.1
export OMP_NUM_THREADS=10

env

mpirun -display-devel-map -n 4 --map-by ppr:2:socket ./bcast_nv 800

date
