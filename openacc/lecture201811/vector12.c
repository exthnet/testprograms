#include <stdio.h>
#include <stdlib.h>

void swap(int *a, int n, int m)
{
  int tmp = a[n];
  a[n] = a[m];
  a[m] = tmp;
}

int main(int argc, char **argv)
{
  int i, n;
  double *v1, *v2;
  int *index;

  if(argc==1){
	n = 10;
  }else{
	n = atoi(argv[1]);
  }
  printf("n = %d\n", n);

  v1 = (double*)malloc(sizeof(double)*n);
  v2 = (double*)malloc(sizeof(double)*n);
  index = (int*)malloc(sizeof(int)*n);
  for(i=0; i<n; i++){
	v1[i] = (double)(i+1);
	v2[i] = 0.0;
	index[i] = i;
  }
  for(i=0; i<n; i++){
	swap(index,rand()%n, rand()%n);
  }
  for(i=0; i<n; i++){
	printf(" %d", index[i]);
  }
  printf("\n");


#pragma acc kernels
  for(i=0; i<n; i++){
	v2[i] = v1[index[i]] * 2.0;
  }

  for(i=0; i<n; i++){
	printf(" %.2f", v2[i]);
  }
  printf("\n");

  free(v1);
  free(v2);
  free(index);

  return 0;
}
