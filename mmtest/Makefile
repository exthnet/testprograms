REAL = -D_DOUBLE
OPTS = -O3
GCC = gcc
GCC_OPTS = -O3
GCC_OPTS_ACC = -O3 -fopenacc -foffload=nvptx-none -foffload="-O3"
IFORT_OPTS = -O0
ICC_OPTS = -O3 -g -qopt-report -qopt-report-phase=vec
OUTDIR = ./out/

PGCC = pgcc
PGCC_OPTS = -fastsse -fopenmp
PGCC_OPTS_ACC = -fastsse -fopenacc

all:
	${GCC} test.c

mm_base: mm_base.c
	${GCC} -Wall ${OPTS} ${REAL} mm_base.c -o ${OUTDIR}/mm_base -msse3 -D_USE_SSE
	${GCC} -Wall ${OPTS} ${REAL} mm_base.c -o ${OUTDIR}/mm_base_mm -msse3 -D_USE_MM_INST -D_USE_SSE
	${GCC} -Wall ${OPTS} ${REAL} mm_base.c -o ${OUTDIR}/mm_base_avx -mavx -D_USE_AVX -D_USE_SSE

mm_base_omp: mm_base.c mm_base_nosimdinst.c
	${GCC} -Wall ${OPTS} ${REAL} mm_base.c -o ${OUTDIR}/mm_base_omp -msse3 -fopenmp -D_USE_SSE
	${GCC} -Wall ${OPTS} ${REAL} mm_base.c -o ${OUTDIR}/mm_base_mm_omp -msse3 -D_USE_MM_INST -fopenmp -D_USE_SSE
	${GCC} -Wall ${OPTS} ${REAL} mm_base.c -o ${OUTDIR}/mm_base_avx_omp -mavx -D_USE_AVX -fopenmp -D_USE_SSE
	${GCC} -Wall ${OPTS} ${REAL} mm_base_nosimdinst.c -o ${OUTDIR}/mm_base_nosimdinst -fopenmp

mm_base_pgi: mm_base_nosimdinst.c
	${PGCC} -Wall ${OPTS} ${REAL} mm_base_nosimdinst.c -o ${OUTDIR}/mm_base_pgi_nosimdinst -fopenmp

mm_base_acc: mm_base_acc.c
	${GCC} -Wall ${GCC_OPTS_ACC} ${REAL} mm_base_acc.c -o ${OUTDIR}/mm_base_acc_gcc
	${PGCC} -Wall ${PGCC_OPTS_ACC} ${REAL} mm_base_acc.c -o ${OUTDIR}/mm_base_acc_pgi

mm_base_icc:
	icc ${ICC_OPTS} -D_DOUBLE mm_base.c -o ${OUTDIR}/mm_base_icc_sse3_novec -xSSE3 -no-vec -D_USE_MM_INST -D_USE_SSE -mkl=sequential
	icc ${ICC_OPTS} -D_DOUBLE mm_base.c -o ${OUTDIR}/mm_base_icc_sse3 -xSSE3 -D_USE_MM_INST -D_USE_SSE -mkl=sequential
	icc ${ICC_OPTS} -D_DOUBLE mm_base.c -o ${OUTDIR}/mm_base_icc_avx_novec -xAVX -no-vec -D_USE_MM_INST -D_USE_AVX -mkl=sequential
	icc ${ICC_OPTS} -D_DOUBLE mm_base.c -o ${OUTDIR}/mm_base_icc_avx -xAVX -D_USE_MM_INST -D_USE_AVX -mkl=sequential

mm_base_icc_omp:
	icc ${ICC_OPTS} -D_DOUBLE mm_base.c -o ${OUTDIR}/mm_base_icc_sse3_novec_omp -xSSE3 -no-vec -D_USE_MM_INST -D_USE_SSE -openmp -mkl=parallel
	icc ${ICC_OPTS} -D_DOUBLE mm_base.c -o ${OUTDIR}/mm_base_icc_sse3_omp -xSSE3 -D_USE_MM_INST -D_USE_SSE -openmp -mkl=parallel
	icc ${ICC_OPTS} -D_DOUBLE mm_base.c -o ${OUTDIR}/mm_base_icc_avx_novec_omp -xAVX -no-vec -D_USE_MM_INST -D_USE_AVX -openmp -mkl=parallel
	icc ${ICC_OPTS} -D_DOUBLE mm_base.c -o ${OUTDIR}/mm_base_icc_avx_omp -xAVX -D_USE_MM_INST -D_USE_AVX -openmp -mkl=parallel
	icc ${ICC_OPTS} -D_DOUBLE mm_base.c -o ${OUTDIR}/mm_base_icc_avx_novec_cr_omp -xAVX -no-vec -D_USE_MM_INST -D_USE_AVX -openmp -mkl=parallel -D_USE_CONST -D_USE_RESTRICT
	icc ${ICC_OPTS} -D_DOUBLE mm_base.c -o ${OUTDIR}/mm_base_icc_avx_cr_omp -xAVX -D_USE_MM_INST -D_USE_AVX -openmp -mkl=parallel -D_USE_CONST -D_USE_RESTRICT
	icc ${ICC_OPTS} -D_DOUBLE mm_base.c -o ${OUTDIR}/mm_base_icc_avx_omp_nomicmkl -xAVX -D_USE_MM_INST -D_USE_AVX -openmp -mkl=parallel -D_NO_MIC_MKL

mm_base_icc_knl:
	icc -O3 -qopenmp -xMIC-AVX512 -D_DOUBLE mm_base.c -o ${OUTDIR}/mm_base_icc_knl_seq -mkl=sequential
	icc -O3 -qopenmp -xMIC-AVX512 -D_DOUBLE mm_base.c -o ${OUTDIR}/mm_base_icc_knl_par -mkl=parallel

mm_base_mic:
	icc -mmic ${ICC_OPTS} -D_DOUBLE mm_base.c -o ${OUTDIR}/mm_base_icc_mic_novec -no-vec -D_USE_MM_INST -D_USE_AVX3 -mkl=sequential
	icc -mmic ${ICC_OPTS} -D_DOUBLE mm_base.c -o ${OUTDIR}/mm_base_icc_mic -D_USE_MM_INST -D_USE_AVX3 -mkl=sequential

mm_base_mic_omp:
	icc -mmic ${ICC_OPTS} -D_DOUBLE mm_base.c -o ${OUTDIR}/mm_base_icc_mic_novec_omp -no-vec -D_USE_MM_INST -D_USE_AVX3 -openmp -mkl=parallel
	icc -mmic ${ICC_OPTS} -D_DOUBLE mm_base.c -o ${OUTDIR}/mm_base_icc_mic_omp -D_USE_MM_INST -D_USE_AVX3 -openmp -mkl=parallel
	icc -mmic ${ICC_OPTS} -D_DOUBLE mm_base.c -o ${OUTDIR}/mm_base_icc_mic_novec_cr_omp -no-vec -D_USE_MM_INST -D_USE_AVX3 -openmp -mkl=parallel -D_USE_CONST -D_USE_RESTRICT
	icc -mmic ${ICC_OPTS} -D_DOUBLE mm_base.c -o ${OUTDIR}/mm_base_icc_mic_cr_omp -D_USE_MM_INST -D_USE_AVX3 -openmp -mkl=parallel -D_USE_CONST -D_USE_RESTRICT

mm_base_mic_omp_nomkl:
	icc -mmic ${ICC_OPTS} -D_DOUBLE mm_base.c -o ${OUTDIR}mm_base_icc_mic_novec_omp_nomkl -no-vec -D_USE_MM_INST -D_USE_AVX3 -openmp -D_NO_MKL
	icc -mmic ${ICC_OPTS} -D_DOUBLE mm_base.c -o ${OUTDIR}mm_base_icc_mic_omp_nomkl -D_USE_MM_INST -D_USE_AVX3 -openmp -D_NO_MKL
	icc -mmic ${ICC_OPTS} -D_DOUBLE mm_base.c -o ${OUTDIR}mm_base_icc_mic_novec_cr_omp_nomkl -no-vec -D_USE_MM_INST -D_USE_AVX3 -openmp -D_NO_MKL -D_USE_CONST -D_USE_RESTRICT
	icc -mmic ${ICC_OPTS} -D_DOUBLE mm_base.c -o ${OUTDIR}mm_base_icc_mic_cr_omp_nomkl -D_USE_MM_INST -D_USE_AVX3 -openmp -D_NO_MKL -D_USE_CONST -D_USE_RESTRICT

mm_cuda: mm_cuda.cu
	nvcc ${OPTS} ${REAL} -c time_cpu_struct.c
	nvcc ${OPTS} ${REAL} -c time_gpu_struct.cu
	nvcc ${OPTS} ${REAL} -c mm_cuda.cu -lineinfo #-arch=sm_80
	nvcc ${OPTS} ${REAL} -o ${OUTDIR}/mm_cuda mm_cuda.o time_cpu_struct.o time_gpu_struct.o -lcublas

mm_f: mm1.f90 mm2.f90
	ifort ${IFORT_OPTS} -no-offload -m64 -openmp -mavx -xAVX -mcmodel=medium -cpp -fixed -no-vec -align array64byte mm1.f90 -o ${OUTDIR}/mmf1_omp
	ifort ${IFORT_OPTS} -no-offload -m64 -openmp -mavx -xAVX -mcmodel=medium -cpp -no-vec -align array64byte mm2.f90 -o ${OUTDIR}/mmf2_omp
mm_f_mic: mm1.f90 mm2.f90
	ifort ${IFORT_OPTS} -m64 -openmp -mmic -mcmodel=medium -cpp -fixed -no-vec -align array64byte mm1.f90 -o ${OUTDIR}/mmf1_omp_mic
	ifort ${IFORT_OPTS} -m64 -openmp -mmic -mcmodel=medium -cpp -no-vec -align array64byte mm2.f90 -o ${OUTDIR}/mmf2_omp_mic
mm_f_ofl: mm1_ofl.f90 mm2_ofl.f90
	ifort -offload-build -mavx -xAVX ${IFORT_OPTS} -m64 -openmp -mcmodel=medium -cpp -fixed -no-vec                    mm1_ofl.f90 -o ${OUTDIR}/mmf1_omp_ofl
	ifort -offload-build -mavx -xAVX ${IFORT_OPTS} -m64 -openmp -mcmodel=medium -cpp -fixed -no-vec -align array64byte mm1_ofl.f90 -o ${OUTDIR}/mmf1_omp_ofl_align
	ifort -offload-build -mavx -xAVX ${IFORT_OPTS} -m64 -openmp -mcmodel=medium -cpp -no-vec                           mm2_ofl.f90 -o ${OUTDIR}/mmf2_omp_ofl
	ifort -offload-build -mavx -xAVX ${IFORT_OPTS} -m64 -openmp -mcmodel=medium -cpp -no-vec        -align array64byte mm2_ofl.f90 -o ${OUTDIR}/mmf2_omp_ofl_align

clean:
	/bin/rm -f *~ *.o
	/bin/rm -f mmf1 mmf1_omp mmf2 mmf2_omp
	/bin/rm -f mmf1_omp_mic mmf2_omp_mic
	/bin/rm -f mmf1_omp_ofl mmf2_omp_ofl
	/bin/rm -f mmf1_omp_ofl_align mmf2_omp_ofl_align
