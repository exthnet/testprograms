#!/usr/bin/python
import os, subprocess, string, sys, time

threads=[1,2,4,8]
bins=["mm_base_icc_sse3_novec_omp", "mm_base_icc_sse3_omp", "mm_base_icc_avx_novec_omp", "mm_base_icc_avx_omp", "mm_base_icc_avx_novec_cr_omp", "mm_base_icc_avx_cr_omp", "mm_base_icc_avx_omp_nomicmkl"]
sizes = [128,256,512,1024,2048,4096]
opt=" -loops 10"

cmd = "ulimit -s unlimited;export KMP_AFFINITY=granularity=fine,compact,1,0;"
print cmd

#PATH=/home/ohshima/impi/4.1.0.027/mic/bin:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin:/opt/bin:/home/ohshima/bin
#LD_LIBRARY_PATH=${HOME}/impi/4.1.0.027/mic/lib:${HOME}/xe/compiler/lib/mic

# MKL
kernels=[50]
for b in bins:
    bin = "../out/" + b
    for t in threads:
        cmd = "export MKL_NUM_THREADS=" + str(t)
        print cmd
        for s in sizes:
            for k in kernels:
                cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
                print cmd
                sys.stdout.flush()
