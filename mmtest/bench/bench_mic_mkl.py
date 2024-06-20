#!/usr/bin/python
import os, subprocess, string, sys, time

threads = [16,57,114,171,228]
bin = "../out/mm_base_icc_mic_cr_omp"
sizes = [128,256,512,1024,2048,4096]
opt=" -loops 10"

cmd = "export LD_LIBRARY_PATH=/home/ohshima/intel/composer_xe_2013.3.163/mkl/lib/mic:/home/ohshima/intel/composer_xe_2013.3.163/compiler/lib/mic:${LD_LIBRARY_PATH}"
print cmd
cmd = "export KMP_AFFINITY=granularity=fine"
print cmd
cmd = "ulimit -s unlimited"
print cmd

# MKL
kernels=[50]
for t in threads:
    cmd = "export MKL_NUM_THREADS=" + str(t)
    print cmd
    for s in sizes:
        for k in kernels:
            cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
            print cmd
sys.stdout.flush()
