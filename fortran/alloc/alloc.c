#include <stdio.h>
#include <stdlib.h>

double frand()
{
  return (rand()/(double)RAND_MAX);
}

int main()
{
  int i, j;
  double **d;
  int size[10] = {1,2,3,4,5,6,7,8,9,10};
  // allocate
  d = (double**)malloc(sizeof(double*)*10);
  for(i=0; i<10; i++){
	d[i] = (double*)malloc(sizeof(double)*size[i]);
	for(j=0; j<size[i]; j++){
	  d[i][j] = frand();
	}
  }
  // print
  for(i=0; i<10; i++){
	for(j=0; j<size[i]; j++){
	  printf(" %.2f", d[i][j]);
	}
	printf("\n");
  }
  // calc
  for(i=0; i<10; i++){
	for(j=0; j<size[i]; j++){
	  d[i][j] *= 2.0;
	}
  }
  // print
  for(i=0; i<10; i++){
	for(j=0; j<size[i]; j++){
	  printf(" %.2f", d[i][j]);
	}
	printf("\n");
  }
  // free
  for(i=0; i<10; i++){
	free(d[i]);
  }
  free(d);
  return 0;
}


#if 0

  double **d;
  int size[10] = {1,2,3,4,5,6,7,8,9,10};
  d = (double**)malloc(sizeof(double*)*10);
  for(i=0; i<10; i++){
	d[i] = (double*)malloc(sizeof(double)*size[i]);
  }

#endif
