#/bin/bash
#PJM -L "vnode=1"
#PJM -L "vnode-core=9"
#PJM -L "rscunit=ito-b"
#PJM -L "rscgrp=ito-g-1-dbg"
#PJM -L "elapse=10:00"
#PJM -j
#PJM -S

module load ~/opt/pgi/modulefiles/pgi/18.10
export PGI_ACC_TIME=1
echo ./c21_c
./c21_c
echo ./c21_c_acc
./c21_c_acc
echo ./f21_f
./f21_f
echo ./f21_f_acc
./f21_f_acc