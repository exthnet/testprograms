#!/usr/bin/python
import os, subprocess, string, sys, time
sizes = [128,256,512,1024,2048]
opt=" -loops 10"

# icc, openmp
bin="./mm_base_icc_sse3_novec_omp"
for s in sizes:
    for k in range(0,27):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        subprocess.call(cmd, shell=True)
        time.sleep(1)
# icc, openmp, vec
bin="./mm_base_icc_sse3_omp"
for s in sizes:
    for k in range(0,27):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        subprocess.call(cmd, shell=True)
        time.sleep(1)
# icc, openmp, sse
bin="./mm_base_icc_sse3_novec_omp"
for s in sizes:
    for k in range(27,33):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        subprocess.call(cmd, shell=True)
        time.sleep(1)
# icc, openmp, sse+vec
bin="./mm_base_icc_sse3_omp"
for s in sizes:
    for k in range(27,33):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        subprocess.call(cmd, shell=True)
        time.sleep(1)
# icc, openmp, avx
bin="./mm_base_icc_avx_novec_omp"
kernels=[40,42,44]
for s in sizes:
    for k in kernels:
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        subprocess.call(cmd, shell=True)
        time.sleep(1)
# icc, openmp, avx+vec
bin="./mm_base_icc_avx_omp"
for s in sizes:
    for k in kernels:
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        subprocess.call(cmd, shell=True)
        time.sleep(1)
