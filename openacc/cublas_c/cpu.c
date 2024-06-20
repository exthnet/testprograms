#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
  int i, j, k;
  int N;
  float *A, *B, *C;
  float alpha=1.0f, beta=1.0f;

  N = 8;
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
  printf("A\n");
  for(i=0;i<N;i++){
	for(j=0;j<N;j++){
	  printf(" %8.2f", A[i*N+j]);
	}
	printf("\n");
  }
  printf("B\n");
  for(i=0;i<N;i++) {
	for(j=0;j<N;j++){
	  printf(" %8.2f", B[i*N+j]);
	}
	printf("\n");
  }
  printf("C (before)\n");
  for(i=0;i<N;i++){
	for(j=0;j<N;j++){
	  printf(" %8.2f", C[i*N+j]);
	}
	printf("\n");
  }

  for(i=0; i<N; i++){
	for(j=0; j<N; j++){
	  for(k=0; k<N; k++){
		C[i*N+j] += A[i*N+k] * B[k*N+j];
	  }
	}
  }

  printf("C (after)\n");
  for(i=0;i<N;i++){
	for(j=0;j<N;j++){
	  printf(" %8.2f", C[i*N+j]);
	}
	printf("\n");
  }

  free(A); free(B); free(C);
  return 0;
}
