#!/usr/bin/python
import os, subprocess, string, sys, time
sizes = [128,256,512,1024,2048]
opt=" -loops 10"

# gcc, single
bin="./mm_base"
for s in sizes:
    for k in range(0,27):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        subprocess.call(cmd, shell=True)
        time.sleep(1)
# gcc, single, sse
for s in sizes:
    for k in range(27,34):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        subprocess.call(cmd, shell=True)
        time.sleep(1)
# gcc, single, sse(mm_inst)
bin="./mm_base_mm"
for s in sizes:
    for k in range(27,34):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        subprocess.call(cmd, shell=True)
        time.sleep(1)
# gcc, single, avx
bin="./mm_base_avx"
for s in sizes:
    for k in range(40,46):
        cmd = bin + opt + " -size " + str(s) + " -kernel " + str(k)
        print cmd
        sys.stdout.flush()
        subprocess.call(cmd, shell=True)
        time.sleep(1)
