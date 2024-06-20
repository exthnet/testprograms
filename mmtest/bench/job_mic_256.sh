#!/bin/sh
#PBS -q batch
#PBS -o log_bench_mic_256.txt
#PBS -e log_bench_mic_256.err
#PBS -l nodes=1
cd /home/ohshima/work/gitprojects/testprograms/mmtest/bench


#export SINK_LD_LIBRARY_PATH=/home/ohshima/lib:/opt/intel/composer_xe_2013.0.079/compiler/lib/intel64:/opt/intel/mkl/lib/intel64:/lib64
export SINK_LD_LIBRARY_PATH=/opt/intel/lib/mic:/opt/intel/mkl/lib/mic:/usr/linux-k1om-4.7/linux-k1om/lib64:/opt/intel/mic/myo/lib:/opt/intel/impi/4.1.0.024/mic/lib
ulimit -s unlimited
export LD_LIBRARY_PATH=/opt/intel/composer_xe_2013.0.079/compiler/lib/intel64:/opt/intel/mkl/lib/intel64

/opt/intel/mic/bin/micnativeloadex ./bench_mic_256.sh
