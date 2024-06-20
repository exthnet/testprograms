#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void gpukernel_wrapper(int N, float *C, float *A, float *B);

int main(int argc, char **argv)
{
  int i, N, x;
  float *A, *B, *C;

  x = 10;
  N = 128;
  A = (float*)malloc(sizeof(float)*N);
  B = (float*)malloc(sizeof(float)*N);
  C = (float*)malloc(sizeof(float)*N);

  for(i=0;i<N;i++){
	C[i] = 0.0f;	B[i] = 2.0f;
	A[i] = (float)(i+1)/(float)(N);
  }

  printf("A\n");
  for(i=0; i<N; i++){
	if(i%x==x-1){
	  printf(" %2.4f\n", A[i]);
	}else{
	  printf(" %2.4f", A[i]);
	}
  }
  printf("\n");
  printf("B\n");
  for(i=0; i<N; i++){
	if(i%x==x-1){
	  printf(" %2.4f\n", B[i]);
	}else{
	  printf(" %2.4f", B[i]);
	}
  }
  printf("\n");
  printf("C (before)\n");
  for(i=0; i<N; i++){
	if(i%x==x-1){
	  printf(" %2.4f\n", C[i]);
	}else{
	  printf(" %2.4f", C[i]);
	}
  }
  printf("\n");

#pragma acc enter data copyin(A[0:N],B[0:N],C[0:N])

#pragma acc kernels present(A,B,C)
#pragma acc loop independent
	for(i=0; i<N; i++){
	  C[i] += A[i] * B[i];
	}

#pragma acc host_data use_device(A,B,C)
	{
	  gpukernel_wrapper(N, C, A, B);
	}

#pragma acc exit data copyout(C[0:N])

  printf("C (after)\n");
  for(i=0; i<N; i++){
	if(i%x==x-1){
	  printf(" %2.4f\n", C[i]);
	}else{
	  printf(" %2.4f", C[i]);
	}
  }
  printf("\n");

  free(A); free(B); free(C);
  return 0;
}
