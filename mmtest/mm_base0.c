#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <sys/time.h>

/*
  matrix multiply A = B*C on CPU
*/

#ifdef _DOUBLE
#define real double
#endif
#ifdef _SINGLE
#define real float
#endif

int SIZE = 16;
int RANDSEED = 0;
int LOOPS = 10;
real *g_A, *g_B, *g_C;

real frand()
{
  //return (real)(rand()%1000) / 1000.0;
  return (real)((double)(rand()%10) / 5.0);
}

void cpu_kernel(int size, real *_a, real *_b, real *_c)
{
  int i, j, k;
  for(i=0; i<size; i++){
	for(j=0; j<size; j++){
	  for(k=0; k<size; k++){
		_a[j*size+i] += _b[j*size+k] * _c[k*size+i];
	  }
	}
  }
}

// ******** ******** ******** ********
// main
// ******** ******** ******** ********
int main(int argc, char** argv)
{
  int i;
  if(argc!=2){
	printf("usage: %s size\n", argv[0]);
	return -1;
  }
  SIZE = atoi(argv[1]);
  printf("size %d\n", SIZE);

  int t,g;
  g_A = (real*)malloc(sizeof(real)*SIZE*SIZE);
  g_B = (real*)malloc(sizeof(real)*SIZE*SIZE);
  g_C = (real*)malloc(sizeof(real)*SIZE*SIZE);

  for(g=0; g<SIZE; g++){
	for(t=0; t<SIZE; t++){
	  g_A[g*SIZE+t] = (real)0.0;
	  g_B[g*SIZE+t] = frand();
	  g_C[g*SIZE+t] = frand();
	}
  }

  struct timeval tBegin, tEnd;
  struct timezone tz;
  double *dSecs;
  dSecs = (double*)malloc(sizeof(double)*LOOPS);
  double dBegin, dEnd;
  cpu_kernel(SIZE, g_A, g_B, g_C);
  for(i=0;i<LOOPS;i++){
	gettimeofday(&tBegin, &tz);
	cpu_kernel(SIZE, g_A, g_B, g_C);
	gettimeofday(&tEnd, &tz);
	dBegin= tBegin.tv_sec + (double)tBegin.tv_usec*1.0e-6;
	dEnd= tEnd.tv_sec + (double)tEnd.tv_usec*1.0e-6;
	dSecs[i]= dEnd - dBegin;
  }
  double dSecSum=0.0;
  double dSecMin=999999.0;
  double dSecMax=0.0;
  for(i=0;i<LOOPS;i++){
	if(dSecs[i]<dSecMin)dSecMin=dSecs[i];
	if(dSecs[i]>dSecMax)dSecMax=dSecs[i];
	dSecSum+=dSecs[i];
  }
  printf("performance: size %d^2, total %f sec, average %f sec, min %f sec, max %f sec, %f Mflops\n",
		 SIZE, dSecSum, dSecSum/(double)LOOPS, dSecMin, dSecMax,
		 (double)(LOOPS*2.0*(double)SIZE*(double)SIZE*(double)SIZE)/dSecSum/1000.0/1000.0);

  free(g_A);
  free(g_B);
  free(g_C);

  return 0;
}
