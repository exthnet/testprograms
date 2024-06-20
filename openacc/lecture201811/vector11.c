#include <stdio.h>
#include <stdlib.h>

void sub(int n, double *v)
{
  int i;
#pragma acc kernels
  for(i=0; i<n; i++){
	v[i] = v[i] * 2.0;
  }
}

int main(int argc, char **argv)
{
  int i, n;
  double *v1;

  if(argc==1){
	n = 10;
  }else{
	n = atoi(argv[1]);
  }
  printf("n = %d\n", n);

  v1 = (double*)malloc(sizeof(double)*n);
  for(i=0; i<n; i++){
	v1[i] = (double)(i+1);
  }

  sub(n, v1);

  for(i=0; i<n; i++){
	printf(" %.2f", v1[i]);
  }
  printf("\n");

  free(v1);

  return 0;
}
