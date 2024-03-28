#!/bin/bash -x
c=${OMPI_COMM_WORLD_LOCAL_RANK}
c2=$(( $c * 10 + 9 ))
echo "c=${c}"
echo "c2=${c2}"
export I_MPI_ASYNC_PROGRESS=ON
export I_MPI_ASYNC_PROGRESS_PIN=${c2}
$@
