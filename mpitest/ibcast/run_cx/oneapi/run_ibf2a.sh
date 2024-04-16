#!/bin/bash
#env
#echo ${MPI_LOCALRANKID}
id=$(( (${MPI_LOCALRANKID} + 1) * 10 - 1 ))
export I_MPI_ASYNC_PROGRESS_PIN=${id}
echo export I_MPI_ASYNC_PROGRESS_PIN=${id}
$@
