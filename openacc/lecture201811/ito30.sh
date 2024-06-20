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
echo ./c30_c
./c30_c
echo ./c30_c_acc
./c30_c_acc
echo ./f30_f
./f30_f
echo ./f30_f_acc
./f30_f_acc
