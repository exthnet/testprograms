#!/bin/bash

#PBS -q knsc-q1
#PBS -l select=1:ncpus=20
#PBS -l walltime=6:00:00
#PBS -j oe

date
hostname
cd ${PBS_O_WORKDIR}

export OMP_NUM_THREADS=10
./sgemm 1000
./sgemm 2000
