#!/bin/bash
#$-pe mvapich 96
#$-q itc.q
#$-l h_rt=120
#$-cwd
#$-j y

module avail
module load pgi64/15.5
module load mpich/3.1.3/15.5
unset -f module
export MV2_SMP_USE_CMA=0
export MV2_ENABLE_AFFINITY=0

date
hostname

ulimit -s 65535
export OMP_NUM_THREADS=12

mkdir ${JOB_ID}
cp $TMPDIR/machines ${JOB_ID}/machines
cp $TMPDIR/machines.colon2 ${JOB_ID}/machines.colon2
cp $TMPDIR/machines.colon1 ${JOB_ID}/machines.colon1
mpiexec.hydra -n 8 -env OMP_NUM_THREADS 2 -f $TMPDIR/machines.colon1 ./mpihello_flat_f

date
