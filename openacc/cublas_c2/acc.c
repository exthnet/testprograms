#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <sys/time.h>

int main(int argc, char **argv)
{
  int i, j, k;
  int N;
  float *A, *B, *C;
  float alpha=1.0f, beta=1.0f;
  cublasHandle_t handle;
  struct timeval tv1, tv2;

  if(argc!=2){
	printf("usage: %s N\n", argv[0]);
	return -1;
  }

  N = atoi(argv[1]);
  printf("N = %d\n", N);

  A = (float*)malloc(sizeof(float)*N*N);
  B = (float*)malloc(sizeof(float)*N*N);
  C = (float*)malloc(sizeof(float)*N*N);
  for(i=0;i<N;i++){
	for(j=0;j<N;j++){
	  C[i*N+j] = 0.0f;
	  A[i*N+j] = (float)(i) + (float)j/10.0f;
	  B[i*N+j] = (float)(i) + (float)j/10.0f;
	}
  }

#pragma acc enter data copyin(A[0:N*N], B[0:N*N], C[0:N*N])
  gettimeofday(&tv1,NULL);
#pragma acc kernels
#pragma acc loop independent collapse(2)
  for(i=0; i<N; i++){
	for(j=0; j<N; j++){
#pragma acc loop seq
	  for(k=0; k<N; k++){
		C[i*N+j] += A[i*N+k] * B[k*N+j];
	  }
	}
  }
  gettimeofday(&tv2,NULL);
#pragma acc exit data copyout(C[0:N*N])

  //t2 = omp_get_wtime();
  //printf("TIME: %d %f\n", N, t2-t1);
  printf("TIME: %d %f\n", N, (tv2.tv_sec+tv2.tv_usec*1e-6)-(tv1.tv_sec+tv1.tv_usec*1e-6));

  free(A); free(B); free(C);
  return 0;
}
