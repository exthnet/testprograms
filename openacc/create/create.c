#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
  int i;
  int n = 10;
  double *d;
  d = (double*)malloc(sizeof(double)*n);

#pragma acc kernels loop copyin(d[0:n]) copyout(d[0:n])
  for(i=0; i<n; i++){
	d[i] = (double)i;
  }

  printf("result:");
  for(i=0; i<n; i++){
	printf(" %.2f", d[i]);
  }
  printf("\n");
  free(d);
  return 0;
}
