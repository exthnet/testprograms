#!/bin/bash
#SBATCH -p photon

hostname
date
. /etc/profile.d/modules.sh
module load compiler/2024.0.2 mkl/2024.0 mpi/2021.11
. /home/share/intel/oneapi/vtune/2024.0/env/vars.sh

export I_MPI_DEBUG=5
export OMP_NUM_THREADS=2
export I_MPI_HYDRA_BOOTSTRAP=ssh

mpiexec.hydra -n 2 ./a.out

mpirun -n 2 ./run_vtune.sh ./a.out
vtune --collect hotspots -r result_${SLURM_JOBID}_b mpiexec.hydra -n 2 ./a.out

date

