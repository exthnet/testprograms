#!/bin/bash -x
#PJM -L rscgrp=cx-single
#PJM -L elapse=3:00
#PJM -j

date
hostname

module load hpc_sdk/23.1
export OMP_NUM_THREADS=3

env

n=1000
#mpirun -display-devel-map -n 4 --map-by ppr:2:socket ./bcast_nvf ${n}
#mpirun -display-devel-map -n 4 --map-by ppr:2:socket ./bcast_nvf ${n}
#mpirun -display-devel-map -n 4 --map-by ppr:2:socket ./bcast_nvf ${n}
mpirun -display-devel-map -n 4 --map-by ppr:4:socket:pe=4 ./bcast_nvf ${n}
mpirun -display-devel-map -n 4 --map-by ppr:4:socket:pe=4 ./bcast_nvf ${n}
mpirun -display-devel-map -n 4 --map-by ppr:4:socket:pe=4 ./bcast_nvf ${n}
#mpirun -display-devel-map -n 4 --map-by ppr:4:socket:pe=5 ./bcast_nvf ${n}
#mpirun -display-devel-map -n 4 --map-by ppr:4:socket:pe=5 ./bcast_nvf ${n}
#mpirun -display-devel-map -n 4 --map-by ppr:4:socket:pe=5 ./bcast_nvf ${n}

date
