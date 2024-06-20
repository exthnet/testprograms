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
echo ./c20_c
./c20_c
echo ./c20_c_acc
./c20_c_acc
echo ./f20_f
./f20_f
echo ./f20_f_acc
./f20_f_acc
