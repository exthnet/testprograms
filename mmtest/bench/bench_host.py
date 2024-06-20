#!/usr/bin/python
import os, subprocess, string, sys, time
#sizes = [128,256,512,1024,2048]
sizes = [4096]
opt=" -loops 10"
#for s in range(40,1001,40):
#    for k in range(0,3):

#cmd = "export OMP_NUM_THREADS="
#print cmd

# gcc, single
bin="../out/mm_base"
for s in sizes:
    for k in range(0,27):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        #print cmd
        #sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)
# gcc, single, sse
for s in sizes:
    for k in range(27,34):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        #print cmd
        #sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)
# gcc, single, sse(mm_inst)
bin="../out/mm_base_mm"
for s in sizes:
    for k in range(27,34):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        #print cmd
        #sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)
# gcc, single, avx
bin="../out/mm_base_avx"
for s in sizes:
    for k in range(40,46):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        #print cmd
        #sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)

# gcc, openmp
bin="../out/mm_base_omp"
for s in sizes:
    for k in range(0,27):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        #print cmd
        #sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)
# gcc, openmp, sse
for s in sizes:
    for k in range(27,33):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        #print cmd
        #sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)
# gcc, openmp, sse(mm_inst)
bin="../out/mm_base_mm_omp"
for s in sizes:
    for k in range(27,33):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        #print cmd
        #sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)
# gcc, openmp, avx
bin="../out/mm_base_avx_omp"
kernels=[40,42,44]
for s in sizes:
    for k in kernels:
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        #print cmd
        #sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)

# icc, single
bin="../out/mm_base_icc_sse3_novec"
for s in sizes:
    for k in range(0,27):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        #print cmd
        #sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)
# icc, single, vec
bin="../out/mm_base_icc_sse3"
for s in sizes:
    for k in range(0,27):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        #print cmd
        #sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)
# icc, single, sse
bin="../out/mm_base_icc_sse3_novec"
for s in sizes:
    for k in range(27,34):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        #print cmd
        #sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)
# icc, single, sse+vec
bin="../out/mm_base_icc_sse3"
for s in sizes:
    for k in range(27,34):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        #print cmd
        #sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)
# icc, single, avx
bin="../out/mm_base_icc_avx_novec"
for s in sizes:
    for k in range(40,46):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        #print cmd
        #sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)
# icc, single, avx+vec
bin="../out/mm_base_icc_avx"
for s in sizes:
    for k in range(40,46):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        #print cmd
        #sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)



## ICC vec
# icc, openmp, vec
bin="../out/mm_base_icc_sse3_omp"
for s in sizes:
    for k in range(0,27):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)
# icc, openmp, sse+vec
bin="../out/mm_base_icc_sse3_omp"
for s in sizes:
    for k in range(27,33):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)
# icc, openmp, avx+vec
bin="../out/mm_base_icc_avx_omp"
kernels=[40,42,44]
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

## ICC novec
# icc, openmp
bin="../out/mm_base_icc_sse3_novec_omp"
for s in sizes:
    for k in range(0,27):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)
# icc, openmp, sse
bin="../out/mm_base_icc_sse3_novec_omp"
for s in sizes:
    for k in range(27,33):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        #subprocess.call(cmd, shell=True)
        #time.sleep(1)
# icc, openmp, avx
bin="../out/mm_base_icc_avx_novec_omp"
kernels=[40,42,44]
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
