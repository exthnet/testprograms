#!/bin/sh
#------pjsub option ------#
#PJM -L         "rscgrp=short"
#PJM -L         "node=1"
#PJM -L         "elapse=10:00"
#PJM -g gc26
#------Proguram exection ------#

date
cd ${PJM_O_WORKDIR}
export OMP_NUM_THREADS=16

../out/mm_fx10 -size 1000 -kernel 51
