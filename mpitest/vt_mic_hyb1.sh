#!/bin/sh

#PBS -q knsc-h1
#PBS -l select=1:ncpus=20:host=cn01+1:ncpus=20:host=cn02
#PBS -l walltime=00:10:00

date
hostname
which mpirun
cd ${PBS_O_WORKDIR}
cat ${PBS_NODEFILE}

export I_MPI_ENV_PREFIX_LIST=ivb:IVB,knc:MIC
export IVB_OMP_NUM_THREADS=1
export MIC_OMP_NUM_THREADS=4
export IVB_KMP_AFFINITY=granularity=fine,compact
export MIC_KMP_AFFINITY=granularity=fine,balanced
export I_MPI_MIC=enable

echo "======== ======== ======== ========"
cp ${PBS_NODEFILE} ./hosts.${PBS_JOBID}.txt
sed "s/$/-mic0/" ./hosts.${PBS_JOBID}.txt > ./michosts.${PBS_JOBID}.txt
#mpirun -f ./hosts.${PBS_JOBID}.txt -n 2 -ppn 1 amplxe-cl -collect=general-exploration -target-system=mic-host-launch -result-dir=results.${PBS_JOBID} -analyze-system -duration 180 &
#-start-paused &
mpirun -f ./hosts.${PBS_JOBID}.txt -n 2 -ppn 1 amplxe-cl -collect=advanced-hotspots -target-system=mic-host-launch -result-dir=results.${PBS_JOBID} -analyze-system -duration 300 &
mpirun -n 2 -ppn 1 -f ./michosts.${PBS_JOBID}.txt ./hybhello1.mic
wait

date
