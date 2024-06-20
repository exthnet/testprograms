#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
  int i, j, l;
  int m; // row of A & C
  int n; // col of B & C
  int k; // col of A, row of B
  double **a=NULL, **b=NULL, **c=NULL;
  int out;

  out = 1;
  if(argc<4){
	printf("usage: %s m n k (out)\n", argv[0]);
	return -1;
  }else{
	m = atoi(argv[1]);
	n = atoi(argv[2]);
	k = atoi(argv[3]);
	if(argc>=5){
	  out = atoi(argv[4]);
	}
  }
  printf("m,n,k = %d,%d,%d\n", m,n,k);

  a = (double**)malloc(sizeof(double*)*m); for(i=0;i<m;i++)a[i] = (double*)malloc(sizeof(double)*k);
  b = (double**)malloc(sizeof(double*)*k); for(i=0;i<k;i++)b[i] = (double*)malloc(sizeof(double)*n);
  c = (double**)malloc(sizeof(double*)*m); for(i=0;i<m;i++)c[i] = (double*)malloc(sizeof(double)*n);
  for(i=0; i<m; i++){
	for(j=0; j<k; j++){
	  a[i][j] = (double)(i+1) + (double)(j+1)*0.01;
	}
  }
  for(i=0; i<k; i++){
	for(j=0; j<n; j++){
	  b[i][j] = (double)(i+1) + (double)(j+1)*0.01;
	}
  }
  for(i=0; i<m; i++){
	for(j=0; j<n; j++){
	  c[i][j] = 0.0;
	}
  }
  if(out!=0){
  printf("A:\n");
  for(i=0; i<m; i++){
	for(j=0; j<k; j++){
	  printf(" %.2f", a[i][j]);
	}
	printf("\n");
  }
  printf("B:\n");
  for(i=0; i<k; i++){
	for(j=0; j<n; j++){
	  printf(" %.2f", b[i][j]);
	}
	printf("\n");
  }
  }

#pragma acc kernels
#pragma acc loop gang vector collapse(2)
  for(i=0; i<m; i++){
	for(j=0; j<n; j++){
	  for(l=0; l<k; l++){
		c[i][j] += a[i][l] * b[l][j];
	  }
	}
  }

  if(out!=0){
	printf("result:\n");
	for(i=0; i<m; i++){
	  for(j=0; j<n; j++){
		printf(" %.2f", c[i][j]);
	  }
	  printf("\n");
	}
	printf("\n");
  }

  for(i=0; i<m; i++)free(a[i]); free(a);
  for(i=0; i<k; i++)free(b[i]); free(b);
  for(i=0; i<m; i++)free(c[i]); free(c);

  return 0;
}
