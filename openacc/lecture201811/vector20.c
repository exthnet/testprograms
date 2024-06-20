#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
  int i, j, n=10;
  double v1[10];

  for(i=0; i<n; i++){
	v1[i] = (double)(i+1);
  }

#pragma acc data copy(v1)
  {
#pragma acc kernels
	for(i=0; i<n; i++){
	  v1[i] = v1[i] * 2.0;
	}

	for(i=0; i<n; i++){
	  printf(" %.2f", v1[i]);
	}
	printf("\n");

#pragma acc kernels
	for(i=0; i<n; i++){
	  v1[i] = v1[i] * 2.0;
	}
  }

  for(i=0; i<n; i++){
	printf(" %.2f", v1[i]);
  }
  printf("\n");

  return 0;
}
