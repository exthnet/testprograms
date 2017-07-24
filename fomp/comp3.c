#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void func(double *x, double *y)
{
  int tid;
  double *p=NULL;
  tid = omp_get_thread_num();
  printf("tid %d\n", tid);
#pragma omp barrier
  if(tid==0){
	p = x;
  }
#pragma omp barrier
  if(tid==1){
	p = y;
  }
#pragma omp barrier

#pragma omp critical
  {
	printf("%d %f\n",tid,p[0]);
  }
}

int main()
{
  double *x, *y;
  int i, n;
  n = 10;
  x = (double*)malloc(sizeof(double)*n);
  y = (double*)malloc(sizeof(double)*n);

  for(i=0; i<n; i++){
	x[i] = 1.0;
	y[i] = 2.0;
  }

#pragma omp parallel
  {
	func(x,y);
  }

  free(x);
  free(y);
  return 0;
}

