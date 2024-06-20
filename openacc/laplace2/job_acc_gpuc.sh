#!/bin/sh
##$-q hpci.q@@hpci
#$-cwd
#$-j y
date
hostname
cd ${SGE_O_WORKDIR}
export PGI_ACC_NOTIFY=15
./laplace2acc-F
date
