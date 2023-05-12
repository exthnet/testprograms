CC = mpicc
FC = mpif90
CFLAGS = -mp=numa -fastsse -Minfo -Mmpi=mpich -mcmodel=medium
FFLAGS = -mp=numa -fastsse -Minfo -Mmpi=mpich -mcmodel=medium
RM = rm -f

all:
	${CC} ${CFLAGS} ../mpihello.c -o ./mpihello_c
	${FC} ${FFLAGS} ../mpihello.f -o ./mpihello_f
