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
echo c
./v0c_c
echo c_acc
./v0c_c_acc
echo f
./v0f_f
echo f_acc
./v0f_f_acc

export PGI_ACC_NOTIFY=3
echo c_acc
./v0c_c_acc
