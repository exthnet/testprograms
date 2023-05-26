#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

double frand(double d)
{
  return (double)rand()/(double)RAND_MAX*d;
}

int main(int argc, char **argv)
{
  double *a, *b, *c;
  int i, j, k;
  int n, slice, iter;
  int rank;
  double t_begin, t_end;
  double t_comm1, t_comm2, t_comm;
  double t_calc1, t_calc2, t_calc;
  double tmpsum;

  slice = 10;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("rank=%d\n", rank);
  if(argc==1){
	n = 10;
  }else{
	n = atoi(argv[1]);
  }
  printf("n=%d\n", n);
  a = (double*)malloc(sizeof(double)*n*n*slice);
  b = (double*)malloc(sizeof(double)*n*n*slice);
  c = (double*)malloc(sizeof(double)*n*n*slice);

  srand(0);
  if(rank==0){
	for(i=0; i<n*n*slice; i++)a[i]=frand(1.0);
	for(i=0; i<n*n*slice; i++)b[i]=frand(1.0);
	for(i=0; i<n*n*slice; i++)c[i]=0.0;
  }

  t_comm = 0.0;
  t_calc = 0.0;

  MPI_Barrier(MPI_COMM_WORLD);
  t_begin = MPI_Wtime();
  for(iter=0; iter<slice; iter++){
	t_comm1 = MPI_Wtime();
	MPI_Bcast(&a[n*n*iter], n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&b[n*n*iter], n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&c[n*n*iter], n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	t_comm2 = MPI_Wtime();
	t_comm += t_comm2-t_comm1;
	t_calc1 = MPI_Wtime();
#pragma omp parallel for collapse(2)
	for(i=0; i<n; i++){
	  for(j=0; j<n; j++){
		c[i*n+j+n*n*iter] += a[i*n+j+n*n*iter] * b[i*n+j+n*n*iter];
	  }
	}
	t_calc2 = MPI_Wtime();
	t_calc += t_calc2-t_calc1;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  t_end = MPI_Wtime();
  tmpsum = 0.0;
  for(i=0; i<n*n*slice; i++)tmpsum += c[i];
  printf("%d result: sum= %lf\n", rank, tmpsum);
  printf("%d result: time= %lf, %lf, %lf\n", rank, t_end-t_begin, t_comm, t_calc);
  free(a);
  free(b);
  free(c);
  MPI_Finalize();
  return 0;
}
