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
echo ./c12_c
./c12_c
echo ./c12_c_acc
./c12_c_acc
echo ./f12_f
./f12_f
echo ./f12_f_acc
./f12_f_acc
