#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
  int size = 10;
  double *a, *b, *c, *d;
  FILE *F;
  int i, j, k;

  a = (double*)malloc(sizeof(double)*size*size);
  b = (double*)malloc(sizeof(double)*size*size);
  c = (double*)malloc(sizeof(double)*size*size);
  d = (double*)malloc(sizeof(double)*size*size);

  for(i=0; i<size; i++){
    for(j=0; j<size; j++){
      a[i*n+j] = (double)i+(double)j/1000.0;
      b[i*n+j] = (double)i+(double)j/1000.0;
      c[i*n+j] = 0.0;
      d[i*n+j] = 0.0;
    }
  }

  for(i=0; i<size; i++){
    for(j=0; j<size; j++){
      for(k=0; k<size; k++){
	c[i*n+j] += a[i*n+k] * b[k*n+j];
      }
    }
  }

  cblas_dgemm
    (
     CblasRowMajor, CblasNoTrans, CblasNoTrans,
     size, size, size, 1.0,
     matrixA, dim,
     matrixB, dim, 0.0, matrixD, size
     );

  F = fopen("c.txt", "w");
  for(i=0; i<size; i++){
    for(j=0; j<size; j++){
      fprintf(F, " %f", c[i*n+j]);
    }
    fprintf(F, "\n");
  }
  fclose(F);
  F = fopen("d.txt", "w");
  for(i=0; i<size; i++){
    for(j=0; j<size; j++){
      fprintf(F, " %f", d[i*n+j]);
    }
    fprintf(F, "\n");
  }
  fclose(F);

  free(a);
  free(b);
  free(c);
  return 0;
}
