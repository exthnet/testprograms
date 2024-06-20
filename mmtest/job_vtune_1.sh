#!/bin/bash
#SBATCH -p photon
hostname
date
. /etc/profile.d/modules.sh
module load compiler/2024.0.2 mkl/2024.0 mpi/2021.11
. /home/share/intel/oneapi/vtune/2024.0/env/vars.sh

./sgemm 1000

vtune -collect hotspots -r ./result_sgemm_${SLURM_JOBID} ./sgemm 1000

