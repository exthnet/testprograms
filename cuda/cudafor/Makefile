tmp:
	nvfortran -gopt -cuda -Minfo=all -gpu=cc70,managed -o struct struct.cuf
	nvfortran -gopt -cuda -Minfo=all -gpu=cc70,managed -o struct0 struct0.cuf

all:
	nvfortran -mp -cuda -Minfo=all -cudalib=cublas,cusolver -gpu=cc80,managed -o stream_cc80 stream.cuf
	nvfortran -mp -cuda -Minfo=all -cudalib=cublas,cusolver -gpu=ccall,managed -o stream_ccall stream.cuf
	mpifort -mp -cuda -Minfo=all -cudalib=cublas,cusolver -gpu=cc80,managed -o hybrid_cc80 hybrid.cuf
	mpifort -mp -cuda -Minfo=all -cudalib=cublas,cusolver -gpu=ccall,managed -o hybrid_ccall hybrid.cuf

test:
	nvfortran -mp -cuda -Minfo -cudalib=cublas,cusolver -gpu=cuda11.0 -o stream_11.0 stream.cuf
	nvfortran -mp -cuda -Minfo -cudalib=cublas,cusolver -gpu=cuda11.8 -o stream_11.8 stream.cuf
	nvfortran -mp -cuda -Minfo -cudalib=cublas,cusolver -gpu=cuda12.0 -o stream_12.0 stream.cuf

gemm_managed:
	nvfortran -mp -cuda -Minfo -Mprof -cudalib=cublas,cusolver -gpu=ccall,managed -o gemm_managed gemm_managed.cuf
	nvfortran -mp -cuda -Minfo -Mprof -cudalib=cublas,cusolver -gpu=ccall,managed -o gemm_managed2 gemm_managed2.cuf
	nvfortran -mp -cuda -Minfo -Mprof -cudalib=cublas,cusolver -gpu=ccall,managed -o gemm_managed3 gemm_managed3.cuf
	nvfortran -mp -cuda -Minfo -Mprof -cudalib=cublas,cusolver -gpu=ccall,managed -o gemm_managed4 gemm_managed4.cuf

exec:
	nvfortran -cpp -mp -cuda -Minfo -Mprof -cudalib=cublas,cusolver -gpu=ccall,managed -o exec_managed3 exec_managed3.cuf
exec4:
	nvfortran -cpp -mp -cuda -Minfo -Mprof -cudalib=cublas,cusolver -gpu=ccall,managed -o exec_managed4 exec_managed4.cuf
	nvfortran -cpp -mp -cuda -Minfo -Mprof -cudalib=cublas,cusolver -gpu=ccall,managed -o exec_managed4a exec_managed4a.cuf
	nvfortran -cpp -mp -cuda -Minfo -Mprof -cudalib=cublas,cusolver -gpu=ccall,managed -o exec_managed4b exec_managed4b.cuf
	nvfortran -cpp -mp -cuda -Minfo -Mprof -cudalib=cublas,cusolver -gpu=ccall,managed -o exec_managed4c exec_managed4c.cuf
exec5:
	nvfortran -cpp -mp -cuda -Minfo -Mprof -cudalib=cublas,cusolver -gpu=ccall,managed -o exec_managed5 exec_managed5.cuf
	nvfortran -cpp -mp -cuda -Minfo -Mprof -cudalib=cublas,cusolver -gpu=ccall,managed -o exec_managed5a exec_managed5a.cuf
	nvfortran -cpp -mp -cuda -Minfo -Mprof -cudalib=cublas,cusolver -gpu=ccall,managed -o exec_managed5b exec_managed5b.cuf
	nvfortran -cpp -mp -cuda -Minfo -Mprof -cudalib=cublas,cusolver -gpu=ccall,managed -o exec_managed5c exec_managed5c.cuf
exec6:
	nvfortran -cpp -mp -cuda -Minfo -Mprof -cudalib=cublas,cusolver -gpu=ccall,managed -o exec_managed6 exec_managed6.cuf
