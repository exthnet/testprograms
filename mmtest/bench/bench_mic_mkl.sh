export LD_LIBRARY_PATH=/home/ohshima/intel/composer_xe_2013.3.163/mkl/lib/mic:/home/ohshima/intel/composer_xe_2013.3.163/compiler/lib/mic:${LD_LIBRARY_PATH}
export KMP_AFFINITY=granularity=fine
ulimit -s unlimited
export MKL_NUM_THREADS=16
../out/mm_base_icc_mic_cr_omp -loops 10 -size 128 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 256 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 512 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 1024 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 2048 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 4096 -kernel 50
export MKL_NUM_THREADS=57
../out/mm_base_icc_mic_cr_omp -loops 10 -size 128 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 256 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 512 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 1024 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 2048 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 4096 -kernel 50
export MKL_NUM_THREADS=114
../out/mm_base_icc_mic_cr_omp -loops 10 -size 128 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 256 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 512 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 1024 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 2048 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 4096 -kernel 50
export MKL_NUM_THREADS=171
../out/mm_base_icc_mic_cr_omp -loops 10 -size 128 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 256 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 512 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 1024 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 2048 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 4096 -kernel 50
export MKL_NUM_THREADS=228
../out/mm_base_icc_mic_cr_omp -loops 10 -size 128 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 256 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 512 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 1024 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 2048 -kernel 50
../out/mm_base_icc_mic_cr_omp -loops 10 -size 4096 -kernel 50
