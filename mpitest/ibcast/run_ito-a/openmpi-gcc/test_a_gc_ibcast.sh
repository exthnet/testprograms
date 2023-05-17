#!/bin/bash
#PJM -L rscunit=ito-a
#PJM -L rscgrp=ito-ss
#PJM -L elapse=10:00
#PJM -L vnode-core=36
#PJM -j
#PJM -S

module load openmpi/3.1.6-nocuda-gcc9.2.0
export OMP_NUM_THREADS=3
#mpirun -n 4 -display-devel-map --map-by ppr:4:socket ./ibcast_gc 1000
#mpirun -n 4 -display-devel-map --map-by ppr:4:socket ./ibcast_gc 1000
#mpirun -n 4 -display-devel-map -npersocket 4 ./ibcast_gc 1000
#mpirun -n 4 -display-devel-map -npersocket 4 ./ibcast_gc 1000
mpirun -n 4 -display-devel-map --map-by ppr:4:socket:pe=2 ./ibcast_gc 1000
mpirun -n 4 -display-devel-map --map-by ppr:4:socket:pe=2 ./ibcast_gc 1000
mpirun -n 4 -display-devel-map --map-by ppr:4:socket:pe=3 ./ibcast_gc 1000
mpirun -n 4 -display-devel-map --map-by ppr:4:socket:pe=3 ./ibcast_gc 1000
mpirun -n 4 -display-devel-map --map-by ppr:4:socket:pe=4 ./ibcast_gc 1000
mpirun -n 4 -display-devel-map --map-by ppr:4:socket:pe=4 ./ibcast_gc 1000

