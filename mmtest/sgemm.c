#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <mkl_cblas.h>

int main(int argc, char **argv)
{
  int n;
  float *a, *b, *c;
  FILE *F;
  int i, j, k;
  struct timeval tv1, tv2;

  if(argc==1){
	printf("usage: %s N\n", argv[0]);
	return -1;
  }else{
	n = atoi(argv[1]);
	printf("n = %d\n", n);
  }

  a = (float*)malloc(sizeof(float)*n*n);
  b = (float*)malloc(sizeof(float)*n*n);
  c = (float*)malloc(sizeof(float)*n*n);

  for(i=0; i<n; i++){
    for(j=0; j<n; j++){
      a[i*n+j] = (float)i+(float)j/1000.0;
      b[i*n+j] = (float)i+(float)j/1000.0;
      c[i*n+j] = 0.0;
    }
  }

  /*
  for(i=0; i<n; i++){
    for(j=0; j<n; j++){
      for(k=0; k<n; k++){
	c[i*n+j] += a[i*n+k] * b[k*n+j];
      }
    }
  }
  */

  gettimeofday(&tv1,NULL);
  cblas_sgemm
    (
     CblasRowMajor, CblasNoTrans, CblasNoTrans,
     n, n, n, 1.0f,
     a, n,
     b, n, 1.0f, c, n
     );
  gettimeofday(&tv2,NULL);

  printf("TIME: %d %f\n", n, (tv2.tv_sec+tv2.tv_usec*1e-6)-(tv1.tv_sec+tv1.tv_usec*1e-6));

  /*
  F = fopen("c.txt", "w");
  for(i=0; i<n; i++){
    for(j=0; j<n; j++){
      fprintf(F, " %f", c[i*n+j]);
    }
    fprintf(F, "\n");
  }
  fclose(F);
  F = fopen("d.txt", "w");
  for(i=0; i<n; i++){
    for(j=0; j<n; j++){
      fprintf(F, " %f", d[i*n+j]);
    }
    fprintf(F, "\n");
  }
  fclose(F);
  */

  free(a);
  free(b);
  free(c);
  return 0;
}
