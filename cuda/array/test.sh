#!/bin/sh

#PJM -L "vnode=1"
#PJM -L "vnode-core=36"
#PJM -L "rscunit=ito-b"
#PJM -L "rscgrp=ito-g-4-dbg"
#PJM -L "elapse=10:00"
#PJM -j
#PJM -S

module load cuda

./vectorAdd
./simpleCUBLAS

exit


MAT=10ts
BIN=../bem-bb-SCM.out

module purge
module load mvapich/gdr-2.2-cuda8.0-intel17
lsmod

export MV2_SHOW_CPU_BINDING=1
export MV2_ENABLE_AFFINITY=0
export MV2_USE_CUDA=1
export MV2_GPUDIRECT_GDRCOPY_LIB=/usr/local/lib64/libgdrapi.so.1.2
export LD_PRELOAD=/home/usr0/m70000a/opt/mvapich2-2.2-gdr/lib64/libmpi.so.12.0.5
export MV2_USE_GPUDIRECT_GDRCOPY=0

export KMP_AFFINITY=granularity=fine,compact
export OMP_NUM_THREADS=18
#export MKL_NUM_THREADS=1
env
date
mpirun -np 1 numactl --localalloc --cpunodebind=1 ${BIN} ./input_${MAT}.pbf 2>&1 | tee log_${MAT}_test.txt
date
