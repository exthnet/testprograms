all:
	nvfortran -mp -cuda -Minfo=all -cudalib=cublas,cusolver -gpu=cc80,managed -o stream_cc80 stream.cuf
	nvfortran -mp -cuda -Minfo=all -cudalib=cublas,cusolver -gpu=ccall,managed -o stream_ccall stream.cuf
	mpifort -mp -cuda -Minfo=all -cudalib=cublas,cusolver -gpu=cc80,managed -o hybrid_cc80 hybrid.cuf
	mpifort -mp -cuda -Minfo=all -cudalib=cublas,cusolver -gpu=ccall,managed -o hybrid_ccall hybrid.cuf

test:
	nvfortran -mp -cuda -Minfo -cudalib=cublas,cusolver -gpu=cuda11.0 -o stream_11.0 stream.cuf
	nvfortran -mp -cuda -Minfo -cudalib=cublas,cusolver -gpu=cuda11.8 -o stream_11.8 stream.cuf
	nvfortran -mp -cuda -Minfo -cudalib=cublas,cusolver -gpu=cuda12.0 -o stream_12.0 stream.cuf

