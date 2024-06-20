NVOPT=	-gencode arch=compute_60,code=\"sm_60,compute_60\"
#	-gencode arch=compute_61,code=\"sm_61,compute_61\" \
#	-gencode arch=compute_70,code=\"sm_70,compute_70\"

# 基本：N個単位の連続アクセス
global1: global1.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<

global2: global2.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<

global4: global4.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<

global8: global8.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<

global16: global16.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
# 一般化、GPUカーネル内で繰り返し
globalX: globalX.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<

# Global Memory実験1
globalX2: globalX2.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
# 追加実験
globalX2a: globalX2a.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
globalX2b: globalX2b.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
globalX2c: globalX2c.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
globalX2d: globalX2d.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
globalX2e: globalX2e.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
globalX2f: globalX2f.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
globalX2g: globalX2g.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
globalX2h: globalX2h.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
X2: globalX2 globalX2a globalX2b globalX2c globalX2d globalX2e globalX2f globalX2g globalX2h

# 縦1000倍
globalX3: globalX3.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
globalX3r: globalX3r.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
globalX3r2: globalX3r2.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
# 横1000倍
globalX4: globalX4.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
globalX4r: globalX4r.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
globalX4r2: globalX4r2.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<


# 1WARP, 32x32, W32 & W1
globalW: globalW.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
globalWv: globalWv.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
# template
globalW2: globalW2.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
globalW2v: globalW2v.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
globalW2n: globalW2n.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<

# nWARP
globalAB: globalAB.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<


globalA32: globalA32.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
globalA32r: globalA32r.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<

globalA64: globalA64.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
globalA64r: globalA64r.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<

globalA96: globalA96.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
globalA96r: globalA96r.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<

globalA128: globalA128.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
globalA128r: globalA128r.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<

globalA160: globalA160.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
globalA160r: globalA160r.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<

globalA192: globalA192.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
globalA192r: globalA192r.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<

globalA224: globalA224.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
globalA224r: globalA224r.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<

globalA256: globalA256.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<
globalA256r: globalA256r.cu
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	${NVOPT} -o $@ $<

r: globalA32 globalA64 globalA96 globalA128 globalA160 globalA192 globalA224 globalA256 \
	globalA32r globalA64r globalA96r globalA128r globalA160r globalA192r globalA224r globalA256r

i:
	nvcc -O3 -ccbin icc -Xcompiler="-O3 -no-prec-div -fp-model fast=2 -xHost -qopenmp" \
	-gencode arch=compute_60,code=\"sm_60,compute_60\" \
	-gencode arch=compute_61,code=\"sm_61,compute_61\" \
	-gencode arch=compute_70,code=\"sm_70,compute_70\" \
	-o coalesced coalesced.cu

clean:
	-/bin/rm ./*~ ./*.o

