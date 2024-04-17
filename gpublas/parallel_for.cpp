#include <stdlib.h>
#include <stdio.h>

#define N 64
//#pragma omp requires unified_shared_memory
int main() {
  int n = N;
  int *a = new int[n];
  int *b = new int[n];

  for(int i = 0; i < n; i++)
    a[i] = 0;
  for(int i = 0; i < n; i++)
    b[i] = i;

  #pragma omp target parallel for
  for(int i = 0; i < n; i++)
    a[i] = b[i];

  for(int i = 0; i < n; i++)
    if(a[i] != i)
      printf("error at %d: expected %d, got %d\n", i, i+1, a[i]);

  for(int i = 0; i < n; i++)
    printf(" %d", a[i]);
  printf("\n");
  for(int i = 0; i < n; i++)
    printf(" %d", b[i]);
  printf("\n");
  return 0;
}
