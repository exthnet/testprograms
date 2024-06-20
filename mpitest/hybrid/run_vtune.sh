#!/bin/bash
. /etc/profile.d/modules.sh
module load compiler/2024.0.2 mkl/2024.0 mpi/2021.11

env

vtune --collect hotspots -r result_${SLURM_JOBID}_${PMI_RANK} $@
