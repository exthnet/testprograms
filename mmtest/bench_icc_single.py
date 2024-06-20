#!/usr/bin/python
import os, subprocess, string, sys, time
sizes = [128,256,512,1024,2048]
opt=" -loops 10"

# icc, single
bin="./mm_base_icc_sse3_novec"
for s in sizes:
    for k in range(0,27):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        subprocess.call(cmd, shell=True)
        time.sleep(1)
# icc, single, vec
bin="./mm_base_icc_sse3"
for s in sizes:
    for k in range(0,27):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        subprocess.call(cmd, shell=True)
        time.sleep(1)
# icc, single, sse
bin="./mm_base_icc_sse3_novec"
for s in sizes:
    for k in range(27,34):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        subprocess.call(cmd, shell=True)
        time.sleep(1)
# icc, single, sse+vec
bin="./mm_base_icc_sse3"
for s in sizes:
    for k in range(27,34):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        subprocess.call(cmd, shell=True)
        time.sleep(1)
# icc, single, avx
bin="./mm_base_icc_avx_novec"
for s in sizes:
    for k in range(40,46):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        subprocess.call(cmd, shell=True)
        time.sleep(1)
# icc, single, avx+vec
bin="./mm_base_icc_avx"
for s in sizes:
    for k in range(40,46):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        subprocess.call(cmd, shell=True)
        time.sleep(1)
