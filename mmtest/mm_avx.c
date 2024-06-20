/* -*- C++ -*- */
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <immintrin.h>

/*
  matrix multiply A = B*C on AVX
*/

#ifdef _DOUBLE
#define real double
#endif
#ifdef _SINGLE
#define real float
#endif

real frand()
{
  //return (real)(rand()%1000) / 1000.0;
  return (real)((double)(rand()%10) / 5.0);
}

#ifdef _DOUBLE
int avx_kernel(int kernel, int size, double *_a, double *_b, double *_c)
{
  double __attribute__((aligned(32))) common32[0xff];
#pragma omp parallel
  {
	int i, j;
	__m256d va, vb, v;
	double __attribute__((aligned(32))) *bvec;
	int id = omp_get_thread_num();
	int nth = omp_get_num_threads();
	int iLen = n/nth;
	if(iLen*nth<n)iLen++;
	int iBegin = iLen*id;
	int iEnd = iBegin+iLen<n?iBegin+iLen:n;
	int iStep = 1;
	bvec = &common32[nth*4+id*4];
	for(i=iBegin; i<iEnd; i+=iStep){
	  double tmp = (double)0.0;
	  double tmp2[4];
	  j=irp[i];
	  v = _mm256_setzero_pd();
	  while(j+4<irp[i+1]){
		/*
		  int j2=j;
		  tmp += val[j2] * vec[icol[j2]]; j2++;
		  tmp += val[j2] * vec[icol[j2]]; j2++;
		  tmp += val[j2] * vec[icol[j2]]; j2++;
		  tmp += val[j2] * vec[icol[j2]]; j2++;
		*/
		bvec[0] = vec[icol[j]]; bvec[1] = vec[icol[j+1]]; bvec[2] = vec[icol[j+2]]; bvec[3] = vec[icol[j+3]];
		if((j||0x11)==0){
		  va = _mm256_load_pd(&val[j]);
		}else{
		  va = _mm256_loadu_pd(&val[j]);
		}
		vb = _mm256_load_pd(bvec);
		v  = _mm256_add_pd(v, _mm256_mul_pd(va, vb));
		j+=4;
	  }
	  /*
		  _mm256_store_pd(tmp2, v);
		    tmp += tmp2[0] + tmp2[1] + tmp2[2] + tmp2[3];
	  */
	  v = _mm256_hadd_pd(v,v);
	  _mm256_store_pd(tmp2, v);
	  tmp += tmp2[0] + tmp2[2];
	  for(; j<irp[i+1]; j++){
		tmp += val[j] * vec[icol[j]];
	  }
	  ans[i] = tmp;
	}
  }
  return 0;
}
#endif


#ifdef _SINGLE
int avx_kernel(int kernel, int size, float *_a, float *_b, float *_c)
{
  return 0;
}
#endif

// ******** ******** ******** ********
// main
// ******** ******** ******** ********
int main(int argc, char** argv)
{
  int i;
  if(checkArgs(argc,argv))return -1;

  printf("size %d\n", SIZE);
  srand(RANDSEED);

  //printf("initialize...");
  int t,g;

  g_A = (double*)malloc(sizeof(double)*SIZE*SIZE);
  g_B = (double*)malloc(sizeof(double)*SIZE*SIZE);
  g_C = (double*)malloc(sizeof(double)*SIZE*SIZE);

  for(g=0; g<SIZE; g++){
	for(t=0; t<SIZE; t++){
	  g_A[g*SIZE+t] = 0.0;
	  g_B[g*SIZE+t] = frand();
	  g_C[g*SIZE+t] = frand();
	}
  }
  //printf("done\n");

  struct timeval tBegin, tEnd;
  struct timezone tz;
  double dSec;
  double dBegin, dEnd;
  gettimeofday(&tBegin, &tz);
  for(i=0;i<LOOPS;i++){avx_kernel(KERNEL, SIZE, g_A, g_B, g_C);}
  gettimeofday(&tEnd, &tz);
  dBegin= tBegin.tv_sec + (double)tBegin.tv_usec*1.0e-6;
  dEnd= tEnd.tv_sec + (double)tEnd.tv_usec*1.0e-6;
  dSec= dEnd - dBegin;
  printf("performance: size %d^2, %f sec, %f Mflops\n", SIZE, dSec, (double)(LOOPS*2.0*(double)SIZE*(double)SIZE*(double)SIZE)/dSec/1000.0/1000.0);

  printf("random sampling:");
  for(i=0; i<10; i++){
	int n = rand()%(SIZE*SIZE);
	printf(" %.2f", g_A[n]);
  }
  printf("\n");

  {
	int i, j;
	FILE *F;
	char filename[16];
	snprintf(filename, 16, "A_k%d.txt", KERNEL);
	F = fopen(filename, "w");
	for(i=0; i<SIZE; i++){
	  for(j=0; j<SIZE; j++){
		fprintf(F, " %.2f", g_A[i*SIZE+j]);
	  }
	  fprintf(F, "\n");
	}
	fclose(F);
  }

  free(g_A);
  free(g_B);
  free(g_C);

  return 0;
}
