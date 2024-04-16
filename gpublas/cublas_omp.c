// -*- c++ -*-
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <cublas_v2.h>

double frand()
{
  //return (double)(rand()%1000) / 1000.0;
  //return (double)((double)(rand()%10) / 5.0);
  return ceil((double)(rand()%10));
}

double *a, *b, *c;

int main(int argc, char** argv)
{
  int i, j, k, n, out=1;
  cublasStatus_t stat;
  cublasHandle_t handle;
  double alpha=1.0, beta=0.0;
  double t1, t2;

  if(argc<2){
	printf("usage: %s n (out)\n", argv[0]);
	return -1;
  }
  n = atoi(argv[1]);
  printf("n = %d\n", n);
  if(argc==3)out=atoi(argv[2]);

  stat = cublasCreate(&handle);
  if(stat != CUBLAS_STATUS_SUCCESS){
	printf("CUBLAS initialization failed\n");
  }

  srand(1);

  a = (double*)malloc(sizeof(double)*n*n);
  b = (double*)malloc(sizeof(double)*n*n);
  c = (double*)malloc(sizeof(double)*n*n);

  for(i=0; i<n; i++){
	for(j=0; j<n; j++){
	  //a[i*n+j] = frand();
	  a[i*n+j] = (double)((i+1)*2+j+1);
	}
  }
  for(i=0; i<n; i++){
	for(j=0; j<n; j++){
	  //b[i*n+j] = frand();
	  b[i*n+j] = (double)((i+1)*2+j+1);
	}
  }
  for(i=0; i<n; i++){
	for(j=0; j<n; j++){
	  c[i*n+j] = (double)0.0;
	}
  }

  if(out==1){
	printf("A\n");
	for(i=0; i<n; i++){
	  for(j=0; j<n; j++){
		printf(" %.2f", a[i*n+j]);
	  }
	  printf("\n");
	}
	printf("B\n");
	for(i=0; i<n; i++){
	  for(j=0; j<n; j++){
		printf(" %.2f", b[i*n+j]);
	  }
	  printf("\n");
	}
  }

  t1 = omp_get_wtime();
#pragma omp target data map(to:a[0:n*n],b[0:n*n]) map(tofrom:c[0:n*n])
  {
#pragma omp target data use_device_addr(a,b,c)
	{
	  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n ,n, n, &alpha, a, n, b, n, &beta, c, n);
	}
  }
  t2 = omp_get_wtime();
  printf("time %12.4g sec\n", t2-t1);

  if(out==1){
	printf("C\n");
	for(i=0; i<n; i++){
	  for(j=0; j<n; j++){
		printf(" %.2f", c[i*n+j]);
	  }
	  printf("\n");
	}
  }

  cublasDestroy(handle);
  free(c);
  free(b);
  free(a);
  return 0;
}

