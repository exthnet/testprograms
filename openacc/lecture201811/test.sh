#!/bin/bash
#PJM -L "rscunit=ito-b" 
#PJM -L "rscgrp=ito-g-1"
#PJM -L "vnode=1"
#PJM -L "vnode-core=9"
#PJM -L "elapse=00:05:00"
#PJM -o out.txt
#PJM -e err.txt

module load ~/opt/pgi/modulefiles/pgi/18.10
pgaccelinfo

