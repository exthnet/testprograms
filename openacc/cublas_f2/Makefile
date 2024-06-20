# -*- Makefile -*-

TARGETS=cpu cublas acc hybrid
all: ${TARGETS}

cpu: cpu.f90
	pgf90 -O3 -o cpu cpu.f90

cublas: cublas.cuf
	pgf90 -O3 -o cublas cublas.cuf

acc: acc.f90
	pgf90 -Mcuda=cc20 -acc -O3 -Minfo -ta=tesla,cc20 -o acc acc.f90

hybrid: hybrid.f90
	pgf90 -Mcuda=cc20 -acc -O3 -Minfo -ta=tesla,cc20 -o hybrid hybrid.f90

#cuda: cuda.cuf
#	pgf90 -Mcuda=cc20 -O3 -Minfo -ta=tesla,cc20 -o cuda cuda.cuf

clean:
	-/bin/rm ./*~ ./*.o ./*.mod
	-/bin/rm ${TARGETS}
