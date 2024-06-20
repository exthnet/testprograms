#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>

#define N 1000

void sub()
{
  int i, j, k;
  double d;
  for(i=0; i<N; i++){
	for(j=0; j<N; j++){
	  for(k=0; k<N; k++){
		d += sqrt(d+1.0);
	  }
	}
  }
  printf("d = %e\n", d);
}

int main(int argc, char **argv)
{
  double t1, t2;
  t1 = omp_get_wtime();
  sub();
  t2 = omp_get_wtime();
  printf("time: %e sec\n", t2-t1);
  return 0;
}
