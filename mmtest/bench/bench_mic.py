#!/usr/bin/python
import os, subprocess, string, sys, time
sizes = [128,256,512,1024,2048]
#sizes = [128]
opt=" -loops 10"

#cmd = export LD_LIBRARY_PATH=/home/ohshima/lib
#print cmd

# icc, single
bin="../out/mm_base_icc_mic_novec"
for s in sizes:
    for k in range(0,27):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        #print cmd
        #sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)
# icc, single, vec
bin="../out/mm_base_icc_mic"
for s in sizes:
    for k in range(0,27):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        #print cmd
        #sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)

## ICC vec
# icc, openmp, vec
bin="../out/mm_base_icc_mic_omp"
for s in sizes:
    for k in range(0,27):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)
# icc, openmp, avx3+vec
bin="../out/mm_base_icc_mic_omp"
kernels=[46]
for s in sizes:
    for k in kernels:
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)
kernels=[50]
for s in sizes:
    for k in kernels:
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)

## ICC novec
# icc, openmp
bin="../out/mm_base_icc_mic_novec_omp"
for s in sizes:
    for k in range(0,27):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)
# icc, openmp, avx3
bin="../out/mm_base_icc_mic_novec_omp"
kernels=[46]
for s in sizes:
    for k in kernels:
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)
# MKL
kernels=[50]
for s in sizes:
    for k in kernels:
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)
