#!/bin/bash
#PJM -L "vnode=1"
#PJM -L "vnode-core=9"
#PJM -L "rscunit=ito-b"
#PJM -L "rscgrp=ito-g-1-dbg"
#PJM -L "elapse=10:00"
#PJM -j
#PJM -S

module load intel/2018.3 cuda/9.1
./globalAr 0 2>&1 | tee log_globalAr.txt
./globalAr 1 2>&1 | tee -a log_globalAr.txt
./globalAr 2 2>&1 | tee -a log_globalAr.txt
