#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>

int main(int argc, char **argv)
{
  double t1, t2;
  t1 = omp_get_wtime();
  sleep(5);
  t2 = omp_get_wtime();
  printf("time: %e sec\n", t2-t1);
  return 0;
}
