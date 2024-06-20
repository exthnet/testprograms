#!/bin/bash
#PJM -L "vnode=1"
#PJM -L "vnode-core=9"
#PJM -L "rscunit=ito-b"
#PJM -L "rscgrp=ito-g-1-dbg"
#PJM -L "elapse=10:00"
#PJM -j
#PJM -S

N=256
module load intel/2018.3 cuda/9.1
./globalA${N}r 0 2>&1 | tee log_globalA${N}r.txt
./globalA${N}r 1 2>&1 | tee -a log_globalA${N}r.txt
./globalA${N}r 2 2>&1 | tee -a log_globalA${N}r.txt
