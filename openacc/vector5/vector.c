#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
  int i;
  int n;
  double *v = NULL;

  if(argc==1){
	n = 10;
  }else{
	n = atoi(argv[1]);
  }
  printf("n = %d\n", n);

  v = (double*)malloc(sizeof(double)*n);
  for(i=0; i<n; i++){
	v[i] = (double)(i+1);
  }

  printf("initial:");
  for(i=0; i<n; i++){
	printf(" %.2f", v[i]);
  }
  printf("\n");

#pragma acc kernels
#pragma acc loop
  for(i=0; i<n; i++){
	v[i] *= 2.0;
  }

  printf("result:");
  for(i=0; i<n; i++){
	printf(" %.2f", v[i]);
  }
  printf("\n");

  free(v);

  return 0;
}
