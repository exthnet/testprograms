#!/bin/sh
##$-q hpci.q@@hpci
#$-cwd
#$-j y
date
hostname
cd ${SGE_O_WORKDIR}
./laplace2-F
date
