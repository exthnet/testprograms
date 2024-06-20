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
echo ./c13_c
./c13_c
echo ./c13_c_acc
./c13_c_acc
echo ./f13_f
./f13_f
echo ./f13_f_acc
./f13_f_acc
