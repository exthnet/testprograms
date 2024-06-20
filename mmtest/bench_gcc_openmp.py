#!/usr/bin/python
import os, subprocess, string, sys, time
sizes = [128,256,512,1024,2048]
opt=" -loops 10"

# gcc, openmp
bin="./mm_base_omp"
for s in sizes:
    for k in range(0,27):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        subprocess.call(cmd, shell=True)
        time.sleep(1)
# gcc, openmp, sse
for s in sizes:
    for k in range(27,33):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        subprocess.call(cmd, shell=True)
        time.sleep(1)
# gcc, openmp, sse(mm_inst)
bin="./mm_base_mm_omp"
for s in sizes:
    for k in range(27,33):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        subprocess.call(cmd, shell=True)
        time.sleep(1)
# gcc, openmp, avx
bin="./mm_base_avx_omp"
kernels=[40,42,44]
for s in sizes:
    for k in kernels:
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        subprocess.call(cmd, shell=True)
        time.sleep(1)
