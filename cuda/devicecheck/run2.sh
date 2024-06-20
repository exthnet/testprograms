#!/bin/bash
ID=$(( ${OMPI_COMM_WORLD_RANK} + 1 ))
GPU=`head -n ${ID} ./mig.txt | tail -n 1`
export CUDA_VISIBLE_DEVICES=${GPU}
echo $ID $GPU
echo $@
$@
