// -*- c++ -*-
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
  int i, j, N;
  double *A, *B, *C;
  double *dA, *dB, *dC;

  N = 4;
  A = (double*)malloc(sizeof(double)*N*N);
  B = (double*)malloc(sizeof(double)*N*N);
  C = (double*)malloc(sizeof(double)*N*N);

  for(i=0; i<N; i++){
	for(j=0; j<N; j++){
	  C[i*N+j] = 0.0f;
	  B[i*N+j] = 2.0f;
	  A[i*N+j] = (double)(i)/(double)(N)*1000.0 + (double)(j)/(double)(N);
	}
  }

  printf("A\n");
  for(i=0; i<N; i++){
	for(j=0; j<N; j++){
	  printf(" %2.4f", A[i*N+j]);
	}
	printf("\n");
  }
  printf("B\n");
  for(i=0; i<N; i++){
	for(j=0; j<N; j++){
	  printf(" %2.4f", B[i*N+j]);
	}
	printf("\n");
  }
  printf("C (before)\n");
  for(i=0; i<N; i++){
	for(j=0; j<N; j++){
	  printf(" %2.4f", C[i*N+j]);
	}
	printf("\n");
  }

  for(i=0; i<N; i++){
	for(j=0; j<N; j++){
	  C[i*N+j] = A[i*N+j] * B[i*N+j];
	}
  }

  printf("C (after)\n");
  for(i=0; i<N; i++){
	for(j=0; j<N; j++){
	  printf(" %2.4f", C[i*N+j]);
	}
	printf("\n");
  }

  free(A); free(B); free(C);
  return 0;
}
