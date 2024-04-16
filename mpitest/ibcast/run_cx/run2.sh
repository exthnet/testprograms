#!/bin/bash -x
c=${PMI_RANK}
c2=$(( $c * 10 + 9 ))
echo "c=${c}"
echo "c2=${c2}"
export I_MPI_ASYNC_PROGRESS=ON
export I_MPI_ASYNC_PROGRESS_PIN=${c2}
$@
