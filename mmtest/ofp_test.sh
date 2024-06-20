#!/bin/bash
#PJM -L "rscgrp=debug-flat"
#PJM -L "node=1"
#PJM -L "elapse=10:00"
#PJM -g gh42
#PJM --omp thread=64
#PJM -j

cd ${PJM_O_WORKDIR}

export I_MPI_PLATFORM_CHECK=disable
export I_MPI_PLATFORM=knl
export OMP_NUM_THREADS=64
export KMP_AFFINITY="granularity=fine,balanced"
export KMP_HW_SUBSET=64c@2,1t

cd ${PJM_O_WORKDIR}
# numactl --membind=1 ./out/mm_cbcg -m 4 -n 2 -k 3 -kernel 0  -loops 1
# numactl --membind=1 ./out/mm_cbcg -m 4 -n 2 -k 3 -kernel 1  -loops 1
# numactl --membind=1 ./out/mm_cbcg -m 4 -n 2 -k 3 -kernel 2  -loops 1
# numactl --membind=1 ./out/mm_cbcg -m 4 -n 2 -k 3 -kernel 50 -loops 1
# numactl --membind=1 ./out/mm_cbcg -m 4 -n 2 -k 3 -kernel 51 -loops 1

ldd ./out/mm_cbcg
# ./out/mm_cbcg -m 4 -n 2 -k 3 -kernel 0  -loops 1
# ./out/mm_cbcg -m 4 -n 2 -k 3 -kernel 1  -loops 1
# ./out/mm_cbcg -m 4 -n 2 -k 3 -kernel 2  -loops 1
# ./out/mm_cbcg -m 4 -n 2 -k 3 -kernel 50 -loops 1
./out/mm_cbcg -m 4 -n 2 -k 3 -kernel 51 -loops 1

