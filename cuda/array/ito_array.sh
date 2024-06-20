#!/bin/sh

#PJM -L "vnode=1"
#PJM -L "vnode-core=36"
#PJM -L "rscunit=ito-b"
#PJM -L "rscgrp=ito-b-staff"
#PJM -L "elapse=5:00"
#PJM -j
#PJM -S

module load cuda9.1 intel/2018.3
#module load mvapich/gdr-2.2-cuda8.0-intel17

./array
./array_icc
