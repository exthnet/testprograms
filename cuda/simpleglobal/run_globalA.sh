#!/bin/bash
#PJM -L "vnode=1"
#PJM -L "vnode-core=9"
#PJM -L "rscunit=ito-b"
#PJM -L "rscgrp=ito-g-1-dbg"
#PJM -L "elapse=10:00"
#PJM -j
#PJM -S

module load intel/2018.3 cuda/9.1
./globalA 0 2>&1 | tee log_globalA.txt
./globalA 1 2>&1 | tee -a log_globalA.txt
./globalA 2 2>&1 | tee -a log_globalA.txt
