#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
  int i, j, k;
  int n;
  double **a=NULL, **b=NULL, **c=NULL;
  int out;

  out = 1;
  if(argc==1){
	n = 10;
  }else{
	n = atoi(argv[1]);
	if(argc==3){
	  out = 0;
	}
  }
  printf("n = %d\n", n);

  a = (double**)malloc(sizeof(double*)*n); for(i=0;i<n;i++)a[i] = (double*)malloc(sizeof(double)*n);
  b = (double**)malloc(sizeof(double*)*n); for(i=0;i<n;i++)b[i] = (double*)malloc(sizeof(double)*n);
  c = (double**)malloc(sizeof(double*)*n); for(i=0;i<n;i++)c[i] = (double*)malloc(sizeof(double)*n);
  for(i=0; i<n; i++){
	for(j=0; j<n; j++){
	  a[i][j] = (double)(i+1) + (double)(j+1)*0.01;
	  b[i][j] = (double)(i+1) + (double)(j+1)*0.01;
	  c[i][j] = 0.0;
	}
  }

#pragma acc kernels
#pragma acc loop independent gang,vector collapse(2)
  for(i=0; i<n; i++){
	for(j=0; j<n; j++){
#pragma acc loop seq
	  for(k=0; k<n; k++){
		c[i][j] += a[i][k] * b[k][j];
	  }
	}
  }

  if(out==1){
	printf("result:\n");
	for(i=0; i<n; i++){
	  for(j=0; j<n; j++){
		printf(" %.2f", c[i][j]);
	  }
	  printf("\n");
	}
	printf("\n");
  }

  for(i=0; i<n; i++)free(a[i]); free(a);
  for(i=0; i<n; i++)free(b[i]); free(b);
  for(i=0; i<n; i++)free(c[i]); free(c);

  return 0;
}
