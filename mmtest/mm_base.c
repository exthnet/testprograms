#pragma warning(disable:161)
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <sys/time.h>
#ifndef _NO_MKL
#include <mkl_cblas.h>
#endif

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3
typedef union
{
  __m128d v;
  double d[2];
}v2df_t;

#ifdef _USE_CONST
#define MYCONST const
#else
#define MYCONST
#endif
#ifdef _USE_RESTRICT
#define MYRESTRICT __restrict__
#else
#define MYRESTRICT
#endif

#ifdef _USE_AVX3
#ifndef __USE_MISC
#define __USE_MISC
#endif
#define _malloc_2M(X) mmap(NULL, (X+4096), PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_SHARED|MAP_HUGETLB|MAP_POPULATE, -1, 0)
#include <sys/mman.h>
#include <immintrin.h> // AVX
typedef union
{
  __m512d v;
  double d[8];
}v8df_t;
#endif
#ifdef _USE_AVX
#include <immintrin.h> // AVX
typedef union
{
  __m256d v;
  double d[4];
}v4df_t;
#endif

#define min(a,b) ((a)<(b)?(a):(b))

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
#ifdef _USE_AVX3
__declspec(align(64)) double *g_A, *g_B, *g_C;
#else
#ifdef _USE_AVX
__declspec(align(32)) double *g_A, *g_B, *g_C;
#else
#ifdef _USE_SSE
__declspec(align(16)) double *g_A, *g_B, *g_C;
#else
double *g_A, *g_B, *g_C;
#endif
#endif
#endif
int KERNEL = 0;
int LOOPS=1;
int BLOCKS=16;
#ifndef BLK_J
//#define BLK_J 8
#define BLK_J 32
//#define BLK_J 64
//#define BLK_J 128
//#define BLK_J 256
#endif
#ifndef BLK_I
#define BLK_I 32
//#define BLK_I 64
//#define BLK_I 128
//#define BLK_I 256
#endif
#ifndef BSIZE
#define BSIZE 1024*64
#endif

FILE *fout=NULL;

int MyPrintf(const char* format,...)
{
  va_list va;
  char str[0xffff];
  va_start(va,format);
  vsnprintf(str,0xffff,format,va);
  if(fout==NULL){
	fprintf(stdout, str);
  }else{
	fprintf(fout, str);
  }
  return 0;
}

int checkArgs(int argc, char** argv)
{
  int i;
  printf("checkArgs: ");
  for(i=1; i<argc; i++){
	if(strcmp(argv[i], "-size")==0){
	  SIZE = atoi(argv[++i]);
	  MyPrintf("[size %d]", SIZE);
	}
	if(strcmp(argv[i], "-kernel")==0){
	  KERNEL = atoi(argv[++i]);
	  MyPrintf("[kernel %d]", KERNEL);
	}
	if(strcmp(argv[i], "-loops")==0){
	  LOOPS = atoi(argv[++i]);
	  MyPrintf("[%d loop(s)]", LOOPS);
	}
	if(strcmp(argv[i], "-blocks")==0){
	  BLOCKS = atoi(argv[++i]);
	  MyPrintf("[%d block(s)]", BLOCKS);
	}
	if(strcmp(argv[i], "-fout")==0){
	  char fstr[0xff];
	  strncpy(fstr, argv[++i], 0xff);
	  printf("[fout %s]", fstr);
	  if(fout!=NULL)fclose(fout);
	  fout=fopen(fstr, "w");
	  MyPrintf("[fout %s]", fstr);
	}
	if(strcmp(argv[i], "-chkargs")==0){
	  int valid = atoi(argv[++i]);
	  if(valid!=0){
		int k;
		MyPrintf("command:");for(k=0;k<argc;k++)MyPrintf(" %s",argv[k]);MyPrintf("\n");
	  }
	}
  }
  MyPrintf("\n");
  return 0;
}

real frand()
{
  //return (real)(rand()%1000) / 1000.0;
  return (real)((double)(rand()%10) / 5.0);
}

// 三重ループ
void cpu_kernel_ijk(int kernel, int size, real *_a, real *_b, real *_c)
{
  int i;
#pragma omp parallel for
  for(i=0; i<size; i++){
	int j;
	for(j=0; j<size; j++){
	  int k;
	  for(k=0; k<size; k++){
		_a[j*size+i] += _b[j*size+k] * _c[k*size+i];
	  }
	}
  }
}
// ループ交換
void cpu_kernel_jik(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size; j++){
	int i;
	for(i=0; i<size; i++){
	  int k;
	  for(k=0; k<size; k++){
		_a[j*size+i] += _b[j*size+k] * _c[k*size+i];
	  }
	}
  }
}
// 一時変数化
void cpu_kernel_jik_tmpsum(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size; j++){
	int i;
	for(i=0; i<size; i++){
	  int k;
	  real sum = (real)0.0;
	  for(k=0; k<size; k++){
		sum += _b[j*size+k] * _c[k*size+i];
	  }
	  _a[j*size+i] += sum;
	}
  }
}
// 最内ループのブロック化
void cpu_kernel_jik_blk(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size; j++){
	int i;
	for(i=0; i<size; i++){
	  int k;
	  real sum = (real)0.0;
	  for(k=0; k<size; k+=BLOCKS){
		int k2, kend;
		kend = (k+BLOCKS>size)?(size-k):BLOCKS;
		for(k2=0; k2<kend; k2++){
		  sum += _b[j*size +k+k2] * _c[(k+k2)*size+i];
		}
	  }
	  _a[j*size+i] += sum;
	}
  }
}
// 最内ループのブロック化・別版
void cpu_kernel_jik_blk2(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size; j++){
	int i;
	for(i=0; i<size; i++){
	  int k;
	  real sum = (real)0.0;
	  for(k=0; k<size; k+=BLOCKS){
		int k2, kend;
		kend = (k+BLOCKS>size)?(size):(k+BLOCKS);
		for(k2=k; k2<kend; k2++){
		  sum += _b[j*size +k2] * _c[k2*size+i];
		}
	  }
	  _a[j*size+i] += sum;
	}
  }
}
// 中間ループもブロック化
void cpu_kernel_jik_blk3(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size; j++){
	int i, i2;
	for(i=0; i<size; i+=BLOCKS){
	  int iend;
	  iend = (i+BLOCKS>size)?(size):(i+BLOCKS);
	  for(i2=i; i2<iend; i2++){
		int k;
		real sum = (real)0.0;
		for(k=0; k<size; k+=BLOCKS){
		  int k2, kend;
		  kend = (k+BLOCKS>size)?(size-k):BLOCKS;
		  for(k2=0; k2<kend; k2++){
			sum += _b[j*size +k+k2] * _c[(k+k2)*size+i2];
		  }
		}
		_a[j*size+i2] += sum;
	  }
	}
  }
}
// 最外ループのアンローリング
void cpu_kernel_jik_unroll4(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size-3; j+=4){
	int i;
	for(i=0; i<size; i++){
	  int k;
	  for(k=0; k<size; k++){
		_a[(j+0)*size+i] += _b[(j+0)*size+k] * _c[k*size+i];
		_a[(j+1)*size+i] += _b[(j+1)*size+k] * _c[k*size+i];
		_a[(j+2)*size+i] += _b[(j+2)*size+k] * _c[k*size+i];
		_a[(j+3)*size+i] += _b[(j+3)*size+k] * _c[k*size+i];
	  }
	}
  }
  if(size%4!=0){
    int jbegin = size - (size/4)*4;
	for(j=jbegin; j<size; j++){
	  int i;
	  for(i=0; i<size; i++){
		int k;
		for(k=0; k<size; k++){
		  _a[(j+0)*size+i] += _b[(j+0)*size+k] * _c[k*size+i];
		}
	  }
	}
  }
}
// 最外ループのアンローリング＋一時変数（もちろん高速）
void cpu_kernel_jik_unroll4_tmp(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size-3; j+=4){
	int i;
	for(i=0; i<size; i++){
	  real tmp0 = (real)0.0;
	  real tmp1 = (real)0.0;
	  real tmp2 = (real)0.0;
	  real tmp3 = (real)0.0;
	  int k;
	  for(k=0; k<size; k++){
		tmp0 += _b[(j+0)*size+k] * _c[k*size+i];
		tmp1 += _b[(j+1)*size+k] * _c[k*size+i];
		tmp2 += _b[(j+2)*size+k] * _c[k*size+i];
		tmp3 += _b[(j+3)*size+k] * _c[k*size+i];
	  }
	  _a[(j+0)*size+i] += tmp0;
	  _a[(j+1)*size+i] += tmp1;
	  _a[(j+2)*size+i] += tmp2;
	  _a[(j+3)*size+i] += tmp3;
	}
  }
  if(size%4!=0){
    int jbegin = size - (size/4)*4;
	for(j=jbegin; j<size; j++){
	  int i;
	  for(i=0; i<size; i++){
		real tmp0 = (real)0.0;
		int k;
		for(k=0; k<size; k++){
		  tmp0 += _b[(j+0)*size+k] * _c[k*size+i];
		}
		_a[(j+0)*size+i] += tmp0;
	  }
	}
  }
}
// 最外ループのアンローリング＋一時変数 another（もちろん低速）
void cpu_kernel_jik_unroll4_tmp2(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size-3; j+=4){
	int i;
	for(i=0; i<size; i++){
	  int k;
	  real tmp;
	  tmp = (real)0.0;
	  for(k=0; k<size; k++)tmp += _b[(j+0)*size+k] * _c[k*size+i];
	  _a[(j+0)*size+i] += tmp;
	  tmp = (real)0.0;
	  for(k=0; k<size; k++)tmp += _b[(j+1)*size+k] * _c[k*size+i];
	  _a[(j+1)*size+i] += tmp;
	  tmp = (real)0.0;
	  for(k=0; k<size; k++)tmp += _b[(j+2)*size+k] * _c[k*size+i];
	  _a[(j+2)*size+i] += tmp;
	  tmp = (real)0.0;
	  for(k=0; k<size; k++)tmp += _b[(j+3)*size+k] * _c[k*size+i];
	  _a[(j+3)*size+i] += tmp;
	}
  }
  if(size%4!=0){
    int jbegin = size - (size/4)*4;
	for(j=jbegin; j<size; j++){
	  int i;
	  for(i=0; i<size; i++){
		int k;
		real tmp;
		tmp = (real)0.0;
		for(k=0; k<size; k++)tmp += _b[(j+0)*size+k] * _c[k*size+i];
		_a[(j+0)*size+i] += tmp;
	  }
	}
  }
}
// jkiループ
void cpu_kernel_jki(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size; j++){
	int k;
	for(k=0; k<size; k++){
	  int i;
	  for(i=0; i<size; i++){
		_a[j*size+i] += _b[j*size+k] * _c[k*size+i];
	  }
	}
  }
}

// jkiループ＋最外ループアンローリング
void cpu_kernel_jki_unroll4(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size-3; j+=4){
	int k;
	for(k=0; k<size; k++){
	  int i;
	  for(i=0; i<size; i++){
		_a[(j+0)*size+i] += _b[(j+0)*size+k] * _c[k*size+i];
		_a[(j+1)*size+i] += _b[(j+1)*size+k] * _c[k*size+i];
		_a[(j+2)*size+i] += _b[(j+2)*size+k] * _c[k*size+i];
		_a[(j+3)*size+i] += _b[(j+3)*size+k] * _c[k*size+i];
	  }
	}
  }
  if(size%4!=0){
    int jbegin = size - (size/4)*4;
#pragma omp parallel for
	for(j=jbegin; j<size; j++){
	  int k;
	  for(k=0; k<size; k++){
		int i;
		for(i=0; i<size; i++){
		  _a[(j+0)*size+i] += _b[(j+0)*size+k] * _c[k*size+i];
		}
	  }
	}
  }
}
// jkiループ＋最外ループアンローリング＋一時変数＋共通項まとめ
void cpu_kernel_jki_unroll4_tmp(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size-3; j+=4){
	int i;
	for(i=0; i<size; i++){
	  real tmp0 = (real)0.0;
	  real tmp1 = (real)0.0;
	  real tmp2 = (real)0.0;
	  real tmp3 = (real)0.0;
	  real tmpC;
	  int k;
	  for(k=0; k<size; k++){
		tmpC = _c[k*size+i];
		tmp0 += _b[(j+0)*size+k] * tmpC;
		tmp1 += _b[(j+1)*size+k] * tmpC;
		tmp2 += _b[(j+2)*size+k] * tmpC;
		tmp3 += _b[(j+3)*size+k] * tmpC;
	  }
	  _a[(j+0)*size+i] += tmp0;
	  _a[(j+1)*size+i] += tmp1;
	  _a[(j+2)*size+i] += tmp2;
	  _a[(j+3)*size+i] += tmp3;
	}
  }
  if(size%4!=0){
    int jbegin = size - (size/4)*4;
#pragma omp parallel for
	for(j=jbegin; j<size; j++){
	  int i;
	  for(i=0; i<size; i++){
		real tmp0 = (real)0.0;
		real tmpC;
		int k;
		for(k=0; k<size; k++){
		  tmpC = _c[k*size+i];
		  tmp0 += _b[(j+0)*size+k] * tmpC;
		}
		_a[(j+0)*size+i] += tmp0;
	  }
	}
  }
}

// ijkループ＋最外ループアンローリング
void cpu_kernel_ijk_unroll4(int kernel, int size, real *_a, real *_b, real *_c)
{
  int i;
#pragma omp parallel for
  for(i=0; i<size-3; i+=4){
	int j;
	for(j=0; j<size; j++){
	  int k;
	  for(k=0; k<size; k++){
		_a[j*size+(i+0)] += _b[j*size+k] * _c[k*size+(i+0)];
		_a[j*size+(i+1)] += _b[j*size+k] * _c[k*size+(i+1)];
		_a[j*size+(i+2)] += _b[j*size+k] * _c[k*size+(i+2)];
		_a[j*size+(i+3)] += _b[j*size+k] * _c[k*size+(i+3)];
	  }
	}
  }
  if(size%4!=0){
    int ibegin = size - (size/4)*4;
#pragma omp parallel for
	for(i=ibegin; i<size; i++){
	  int j;
	  for(j=0; j<size; j++){
		int k;
		for(k=0; k<size; k++){
		  _a[j*size+(i+0)] += _b[j*size+k] * _c[k*size+(i+0)];
		}
	  }
	}
  }
}
// ijkループ＋最外ループアンローリング＋一時変数
void cpu_kernel_ijk_unroll4_tmp(int kernel, int size, real *_a, real *_b, real *_c)
{
  int i;
#pragma omp parallel for
  for(i=0; i<size-3; i+=4){
	int j;
	for(j=0; j<size; j++){
	  real tmp0 = (real)0.0;
	  real tmp1 = (real)0.0;
	  real tmp2 = (real)0.0;
	  real tmp3 = (real)0.0;
	  int k;
	  for(k=0; k<size; k++){
		tmp0 += _b[j*size+k] * _c[k*size+(i+0)];
		tmp1 += _b[j*size+k] * _c[k*size+(i+1)];
		tmp2 += _b[j*size+k] * _c[k*size+(i+2)];
		tmp3 += _b[j*size+k] * _c[k*size+(i+3)];
	  }
	  _a[j*size+(i+0)] += tmp0;
	  _a[j*size+(i+1)] += tmp1;
	  _a[j*size+(i+2)] += tmp2;
	  _a[j*size+(i+3)] += tmp3;
	}
  }
  if(size%4!=0){
    int ibegin = size - (size/4)*4;
#pragma omp parallel for
	for(i=ibegin; i<size; i++){
	  int j;
	  for(j=0; j<size; j++){
		int k;
		real tmp0 = (real)0.0;
		for(k=0; k<size; k++){
		  tmp0 += _b[j*size+k] * _c[k*size+(i+0)];
		}
		_a[j*size+(i+0)] += tmp0;
	  }
	}
  }
}
// ijkループ＋最外ループアンローリング＋一時変数＋共通項まとめ
void cpu_kernel_ijk_unroll4_tmp2(int kernel, int size, real *_a, real *_b, real *_c)
{
  int i;
#pragma omp parallel for
  for(i=0; i<size-3; i+=4){
	int j;
	for(j=0; j<size; j++){
	  real tmp0 = (real)0.0;
	  real tmp1 = (real)0.0;
	  real tmp2 = (real)0.0;
	  real tmp3 = (real)0.0;
	  real tmpB;
	  int k;
	  for(k=0; k<size; k++){
		tmpB = _b[j*size+k];
		tmp0 += tmpB * _c[k*size+(i+0)];
		tmp1 += tmpB * _c[k*size+(i+1)];
		tmp2 += tmpB * _c[k*size+(i+2)];
		tmp3 += tmpB * _c[k*size+(i+3)];
	  }
	  _a[j*size+(i+0)] += tmp0;
	  _a[j*size+(i+1)] += tmp1;
	  _a[j*size+(i+2)] += tmp2;
	  _a[j*size+(i+3)] += tmp3;
	}
  }
  if(size%4!=0){
    int ibegin = size - (size/4)*4;
#pragma omp parallel for
	for(i=ibegin; i<size; i++){
	  int j;
	  for(j=0; j<size; j++){
		real tmp0 = (real)0.0;
		real tmpB;
		int k;
		for(k=0; k<size; k++){
		  tmpB = _b[j*size+k];
		  tmp0 += tmpB * _c[k*size+(i+0)];
		}
		_a[j*size+(i+0)] += tmp0;
	  }
	}
  }
}
// jikループ＋最外ループアンローリング＋一時変数＋共通項まとめ＋ポインタ化
void cpu_kernel_jik_unroll4_tmp2_ptr(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size-3; j+=4){
	int i;
	for(i=0; i<size; i++){
	  real tmp0 = (real)0.0;
	  real tmp1 = (real)0.0;
	  real tmp2 = (real)0.0;
	  real tmp3 = (real)0.0;
	  real tmpC;
	  real *tmpB0 = &_b[(j+0)*size];
	  real *tmpB1 = &_b[(j+1)*size];
	  real *tmpB2 = &_b[(j+2)*size];
	  real *tmpB3 = &_b[(j+3)*size];
	  int k;
	  for(k=0; k<size; k++){
		tmpC = _c[k*size+i];
		tmp0 += *tmpB0++ * tmpC;
		tmp1 += *tmpB1++ * tmpC;
		tmp2 += *tmpB2++ * tmpC;
		tmp3 += *tmpB3++ * tmpC;
	  }
	  _a[(j+0)*size+i] += tmp0;
	  _a[(j+1)*size+i] += tmp1;
	  _a[(j+2)*size+i] += tmp2;
	  _a[(j+3)*size+i] += tmp3;
	}
  }
  if(size%4!=0){
    int jbegin = size - (size/4)*4;
#pragma omp parallel for
	for(j=jbegin; j<size; j++){
	  int i;
	  for(i=0; i<size; i++){
		real tmp0 = (real)0.0;
		real tmpC;
		real *tmpB0 = &_b[(j+0)*size];
		int k;
		for(k=0; k<size; k++){
		  tmpC = _c[k*size+i];
		  tmp0 += *tmpB0++ * tmpC;
		}
		_a[(j+0)*size+i] += tmp0;
	  }
	}
  }
}
// register変数化、あまり意味はないと思うのだが、サイズ次第で優位な差が？
void cpu_kernel_jik_unroll4_tmp2_ptr_reg(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size-3; j+=4){
	int i;
	for(i=0; i<size; i++){
	  register real tmp0 = (real)0.0;
	  register real tmp1 = (real)0.0;
	  register real tmp2 = (real)0.0;
	  register real tmp3 = (real)0.0;
	  register real tmpC;
	  real *tmpB0 = &_b[(j+0)*size];
	  real *tmpB1 = &_b[(j+1)*size];
	  real *tmpB2 = &_b[(j+2)*size];
	  real *tmpB3 = &_b[(j+3)*size];
	  int k;
	  for(k=0; k<size; k++){
		tmpC = _c[k*size+i];
		tmp0 += *tmpB0++ * tmpC;
		tmp1 += *tmpB1++ * tmpC;
		tmp2 += *tmpB2++ * tmpC;
		tmp3 += *tmpB3++ * tmpC;
	  }
	  _a[(j+0)*size+i] += tmp0;
	  _a[(j+1)*size+i] += tmp1;
	  _a[(j+2)*size+i] += tmp2;
	  _a[(j+3)*size+i] += tmp3;
	}
  }
  if(size%4!=0){
    int jbegin = size - (size/4)*4;
#pragma omp parallel for
	for(j=jbegin; j<size; j++){
	  int i;
	  for(i=0; i<size; i++){
		register real tmp0 = (real)0.0;
		register real tmpC;
		real *tmpB0 = &_b[(j+0)*size];
		int k;
		for(k=0; k<size; k++){
		  tmpC = _c[k*size+i];
		  tmp0 += *tmpB0++ * tmpC;
		}
		_a[(j+0)*size+i] += tmp0;
	  }
	}
  }
}
// jikループ＋最外ループアンローリング＋一時変数＋共通項まとめ＋ポインタ化＋register＋最内ループアンローリング
void cpu_kernel_jik_unroll4_tmp2_ptr_reg_unroll4(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size-3; j+=4){
	int i;
	for(i=0; i<size; i++){
	  register real tmp0 = (real)0.0;
	  register real tmp1 = (real)0.0;
	  register real tmp2 = (real)0.0;
	  register real tmp3 = (real)0.0;
	  register real tmpC;
	  real *tmpB0 = &_b[(j+0)*size];
	  real *tmpB1 = &_b[(j+1)*size];
	  real *tmpB2 = &_b[(j+2)*size];
	  real *tmpB3 = &_b[(j+3)*size];
	  int k;
	  for(k=0; k+3<size; k+=4){
		tmpC = _c[(k+0)*size+i];
		tmp0 += *tmpB0++ * tmpC;
		tmp1 += *tmpB1++ * tmpC;
		tmp2 += *tmpB2++ * tmpC;
		tmp3 += *tmpB3++ * tmpC;
		tmpC = _c[(k+1)*size+i];
		tmp0 += *tmpB0++ * tmpC;
		tmp1 += *tmpB1++ * tmpC;
		tmp2 += *tmpB2++ * tmpC;
		tmp3 += *tmpB3++ * tmpC;
		tmpC = _c[(k+2)*size+i];
		tmp0 += *tmpB0++ * tmpC;
		tmp1 += *tmpB1++ * tmpC;
		tmp2 += *tmpB2++ * tmpC;
		tmp3 += *tmpB3++ * tmpC;
		tmpC = _c[(k+3)*size+i];
		tmp0 += *tmpB0++ * tmpC;
		tmp1 += *tmpB1++ * tmpC;
		tmp2 += *tmpB2++ * tmpC;
		tmp3 += *tmpB3++ * tmpC;
	  }
	  if(k!=size){
		for(; k<size; k++){
		  tmpC = _c[(k+0)*size+i];
		  tmp0 += *tmpB0++ * tmpC;
		  tmp1 += *tmpB1++ * tmpC;
		  tmp2 += *tmpB2++ * tmpC;
		  tmp3 += *tmpB3++ * tmpC;
		}
	  }
	  _a[(j+0)*size+i] += tmp0;
	  _a[(j+1)*size+i] += tmp1;
	  _a[(j+2)*size+i] += tmp2;
	  _a[(j+3)*size+i] += tmp3;
	}
  }
  if(size%4!=0){
    int jbegin = size - (size/4)*4;
#pragma omp parallel for
	for(j=jbegin; j<size; j++){
	  int i;
	  for(i=0; i<size; i++){
		register real tmp0 = (real)0.0;
		register real tmpC;
		real *tmpB0 = &_b[(j+0)*size];
		int k;
		for(k=0; k<size; k+=4){
		  tmpC = _c[(k+0)*size+i];
		  tmp0 += *tmpB0++ * tmpC;
		  tmpC = _c[(k+1)*size+i];
		  tmp0 += *tmpB0++ * tmpC;
		  tmpC = _c[(k+2)*size+i];
		  tmp0 += *tmpB0++ * tmpC;
		  tmpC = _c[(k+3)*size+i];
		  tmp0 += *tmpB0++ * tmpC;
		}
		if(i!=size){
		  for(; k<size; k++){
			tmpC = _c[(k+0)*size+i];
			tmp0 += *tmpB0++ * tmpC;
		  }
		}
		_a[(j+0)*size+i] += tmp0;
	  }
	}
  }
}
// jikループ＋最外ループアンローリング＋一時変数＋共通項まとめ＋ポインタ化＋register＋最内ループアンローリング＋インクリメント潰し
void cpu_kernel_jik_unroll4_tmp2_ptr_reg_unroll4_noincl(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size-3; j+=4){
	int i;
	for(i=0; i<size; i++){
	  register real tmp0 = (real)0.0;
	  register real tmp1 = (real)0.0;
	  register real tmp2 = (real)0.0;
	  register real tmp3 = (real)0.0;
	  register real tmpC;
	  real *tmpB0 = &_b[(j+0)*size];
	  real *tmpB1 = &_b[(j+1)*size];
	  real *tmpB2 = &_b[(j+2)*size];
	  real *tmpB3 = &_b[(j+3)*size];
	  int k;
	  for(k=0; k+3<size; k+=4){
		tmpC = _c[(k+0)*size+i];
		tmp0 += *tmpB0 * tmpC;
		tmp1 += *tmpB1 * tmpC;
		tmp2 += *tmpB2 * tmpC;
		tmp3 += *tmpB3 * tmpC;
		tmpC = _c[(k+1)*size+i];
		tmp0 += *(tmpB0+1) * tmpC;
		tmp1 += *(tmpB1+1) * tmpC;
		tmp2 += *(tmpB2+1) * tmpC;
		tmp3 += *(tmpB3+1) * tmpC;
		tmpC = _c[(k+2)*size+i];
		tmp0 += *(tmpB0+2) * tmpC;
		tmp1 += *(tmpB1+2) * tmpC;
		tmp2 += *(tmpB2+2) * tmpC;
		tmp3 += *(tmpB3+2) * tmpC;
		tmpC = _c[(k+3)*size+i];
		tmp0 += *(tmpB0+3) * tmpC;
		tmp1 += *(tmpB1+3) * tmpC;
		tmp2 += *(tmpB2+3) * tmpC;
		tmp3 += *(tmpB3+3) * tmpC;
		tmpB0 += 4;
		tmpB1 += 4;
		tmpB2 += 4;
		tmpB3 += 4;
	  }
	  if(k!=size){
		for(; k<size; k++){
		  tmpC = _c[(k+0)*size+i];
		  tmp0 += *tmpB0++ * tmpC;
		  tmp1 += *tmpB1++ * tmpC;
		  tmp2 += *tmpB2++ * tmpC;
		  tmp3 += *tmpB3++ * tmpC;
		}
	  }
	  _a[(j+0)*size+i] += tmp0;
	  _a[(j+1)*size+i] += tmp1;
	  _a[(j+2)*size+i] += tmp2;
	  _a[(j+3)*size+i] += tmp3;
	}
  }
  if(size%4!=0){
    int jbegin = size - (size/4)*4;
#pragma omp parallel for
	for(j=jbegin; j<size; j++){
	  int i;
	  for(i=0; i<size; i++){
		register real tmp0 = (real)0.0;
		register real tmpC;
		real *tmpB0 = &_b[(j+0)*size];
		int k;
		for(k=0; k<size; k+=4){
		  tmpC = _c[(k+0)*size+i];
		  tmp0 += *tmpB0 * tmpC;
		  tmpC = _c[(k+1)*size+i];
		  tmp0 += *(tmpB0+1) * tmpC;
		  tmpC = _c[(k+2)*size+i];
		  tmp0 += *(tmpB0+2) * tmpC;
		  tmpC = _c[(k+3)*size+i];
		  tmp0 += *(tmpB0+3) * tmpC;
		  tmpB0 += 4;
		}
		if(k!=size){
		  for(; k<size; k++){
			tmpC = _c[(k+0)*size+i];
			tmp0 += *tmpB0 * tmpC;
			tmpB0 += 1;
		  }
		}
		_a[(j+0)*size+i] += tmp0;
	  }
	}
  }
}
// tmpCも複製（おそらくほとんど差は無い）
void cpu_kernel_jik_unroll4_tmp2_ptr_reg_unroll4_noincl_clone(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size-3; j+=4){
	int i;
	for(i=0; i<size; i++){
	  register real tmp0 = (real)0.0;
	  register real tmp1 = (real)0.0;
	  register real tmp2 = (real)0.0;
	  register real tmp3 = (real)0.0;
	  register real tmpC0;
	  register real tmpC1;
	  register real tmpC2;
	  register real tmpC3;
	  real *tmpB0 = &_b[(j+0)*size];
	  real *tmpB1 = &_b[(j+1)*size];
	  real *tmpB2 = &_b[(j+2)*size];
	  real *tmpB3 = &_b[(j+3)*size];
	  int k;
	  for(k=0; k+3<size; k+=4){
		tmpC0 = _c[(k+0)*size+i];
		tmp0 += *tmpB0 * tmpC0;
		tmp1 += *tmpB1 * tmpC0;
		tmp2 += *tmpB2 * tmpC0;
		tmp3 += *tmpB3 * tmpC0;
		tmpC1 = _c[(k+1)*size+i];
		tmp0 += *(tmpB0+1) * tmpC1;
		tmp1 += *(tmpB1+1) * tmpC1;
		tmp2 += *(tmpB2+1) * tmpC1;
		tmp3 += *(tmpB3+1) * tmpC1;
		tmpC2 = _c[(k+2)*size+i];
		tmp0 += *(tmpB0+2) * tmpC2;
		tmp1 += *(tmpB1+2) * tmpC2;
		tmp2 += *(tmpB2+2) * tmpC2;
		tmp3 += *(tmpB3+2) * tmpC2;
		tmpC3 = _c[(k+3)*size+i];
		tmp0 += *(tmpB0+3) * tmpC3;
		tmp1 += *(tmpB1+3) * tmpC3;
		tmp2 += *(tmpB2+3) * tmpC3;
		tmp3 += *(tmpB3+3) * tmpC3;
		tmpB0 += 4;
		tmpB1 += 4;
		tmpB2 += 4;
		tmpB3 += 4;
	  }
	  if(k!=size){
		for(; k<size; k++){
		  tmpC0 = _c[(k+0)*size+i];
		  tmp0 += *tmpB0++ * tmpC0;
		  tmp1 += *tmpB1++ * tmpC0;
		  tmp2 += *tmpB2++ * tmpC0;
		  tmp3 += *tmpB3++ * tmpC0;
		}
	  }
	  _a[(j+0)*size+i] += tmp0;
	  _a[(j+1)*size+i] += tmp1;
	  _a[(j+2)*size+i] += tmp2;
	  _a[(j+3)*size+i] += tmp3;
	}
  }
  if(size%4!=0){
    int jbegin = size - (size/4)*4;
#pragma omp parallel for
	for(j=jbegin; j<size; j++){
	  int i;
	  for(i=0; i<size; i++){
		register real tmp0 = (real)0.0;
		register real tmpC0;
		register real tmpC1;
		register real tmpC2;
		register real tmpC3;
		real *tmpB0 = &_b[(j+0)*size];
		int k;
		for(k=0; k+3<size; k+=4){
		  tmpC0 = _c[(k+0)*size+i];
		  tmp0 += *tmpB0 * tmpC0;
		  tmpC1 = _c[(k+1)*size+i];
		  tmp0 += *(tmpB0+1) * tmpC1;
		  tmpC2 = _c[(k+2)*size+i];
		  tmp0 += *(tmpB0+2) * tmpC2;
		  tmpC3 = _c[(k+3)*size+i];
		  tmp0 += *(tmpB0+3) * tmpC3;
		  tmpB0 += 4;
		}
		if(k!=size){
		  for(; k<size; k++){
			tmpC0 = _c[(k+0)*size+i];
			tmp0 += *tmpB0++ * tmpC0;
		  }
		}
		_a[(j+0)*size+i] += tmp0;
	  }
	}
  }
}

// 最内ループの関数化

// jik、最内ループ関数化、外部4x4ブロック化
void cpu_kernel_jik_func_out16_inner_4x4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_func_out16_inner_j4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_func_out16_inner_i4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_func_out16_inner_1(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_func_out16(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size-3; j+=4){
	int i;
	for(i=0; i<size-3; i+=4){
	  cpu_kernel_jik_func_out16_inner_4x4(size,_a,_b,_c,j,i);
	}
	if(i!=size){
	  for(i=i; i<size; i++){
		cpu_kernel_jik_func_out16_inner_j4(size,_a,_b,_c,j,i);
	  }
	}
  }
  if(size%4!=0){
    int jbegin = size - (size/4)*4;
#pragma omp parallel for
	for(j=jbegin; j<size; j++){
	  int i;
	  for(i=0; i<size-3; i+=4){
		cpu_kernel_jik_func_out16_inner_i4(size,_a,_b,_c,j,i);
	  }
	  if(i!=size){
		for(i=i; i<size; i++){
		  cpu_kernel_jik_func_out16_inner_1(size,_a,_b,_c,j,i);
		}
	  }
	}
  }
}
void cpu_kernel_jik_func_out16_inner_4x4(int size, real *_a, real *_b, real *_c, int j, int i)
{
  int k;
  register real tmp0a = (real)0.0;
  register real tmp1a = (real)0.0;
  register real tmp2a = (real)0.0;
  register real tmp3a = (real)0.0;
  register real tmp0b = (real)0.0;
  register real tmp1b = (real)0.0;
  register real tmp2b = (real)0.0;
  register real tmp3b = (real)0.0;
  register real tmp0c = (real)0.0;
  register real tmp1c = (real)0.0;
  register real tmp2c = (real)0.0;
  register real tmp3c = (real)0.0;
  register real tmp0d = (real)0.0;
  register real tmp1d = (real)0.0;
  register real tmp2d = (real)0.0;
  register real tmp3d = (real)0.0;
  register real tmpC0a;
  register real tmpC0b;
  register real tmpC0c;
  register real tmpC0d;
  real *tmpB0 = &_b[(j+0)*size];
  real *tmpB1 = &_b[(j+1)*size];
  real *tmpB2 = &_b[(j+2)*size];
  real *tmpB3 = &_b[(j+3)*size];
  for(k=0; k<size; k++){
	tmpC0a = _c[(k+0)*size+i];
	tmp0a += *tmpB0 * tmpC0a;
	tmp1a += *tmpB1 * tmpC0a;
	tmp2a += *tmpB2 * tmpC0a;
	tmp3a += *tmpB3 * tmpC0a;

	tmpC0b = _c[(k+0)*size+i+1];
	tmp0b += *tmpB0 * tmpC0b;
	tmp1b += *tmpB1 * tmpC0b;
	tmp2b += *tmpB2 * tmpC0b;
	tmp3b += *tmpB3 * tmpC0b;

	tmpC0c = _c[(k+0)*size+i+2];
	tmp0c += *tmpB0 * tmpC0c;
	tmp1c += *tmpB1 * tmpC0c;
	tmp2c += *tmpB2 * tmpC0c;
	tmp3c += *tmpB3 * tmpC0c;

	tmpC0d = _c[(k+0)*size+i+3];
	tmp0d += *tmpB0 * tmpC0d;
	tmp1d += *tmpB1 * tmpC0d;
	tmp2d += *tmpB2 * tmpC0d;
	tmp3d += *tmpB3 * tmpC0d;

	tmpB0 += 1;
	tmpB1 += 1;
	tmpB2 += 1;
	tmpB3 += 1;
  }
  _a[(j+0)*size+i] += tmp0a;
  _a[(j+1)*size+i] += tmp1a;
  _a[(j+2)*size+i] += tmp2a;
  _a[(j+3)*size+i] += tmp3a;
  _a[(j+0)*size+i+1] += tmp0b;
  _a[(j+1)*size+i+1] += tmp1b;
  _a[(j+2)*size+i+1] += tmp2b;
  _a[(j+3)*size+i+1] += tmp3b;
  _a[(j+0)*size+i+2] += tmp0c;
  _a[(j+1)*size+i+2] += tmp1c;
  _a[(j+2)*size+i+2] += tmp2c;
  _a[(j+3)*size+i+2] += tmp3c;
  _a[(j+0)*size+i+3] += tmp0d;
  _a[(j+1)*size+i+3] += tmp1d;
  _a[(j+2)*size+i+3] += tmp2d;
  _a[(j+3)*size+i+3] += tmp3d;
}
void cpu_kernel_jik_func_out16_inner_j4(int size, real *_a, real *_b, real *_c, int j, int i)
{
  int k;
  register real tmp0a = (real)0.0;
  register real tmp1a = (real)0.0;
  register real tmp2a = (real)0.0;
  register real tmp3a = (real)0.0;
  register real tmpC0a;
  real *tmpB0 = &_b[(j+0)*size];
  real *tmpB1 = &_b[(j+1)*size];
  real *tmpB2 = &_b[(j+2)*size];
  real *tmpB3 = &_b[(j+3)*size];
  for(k=0; k<size; k++){
	tmpC0a = _c[(k+0)*size+i];
	tmp0a += *tmpB0 * tmpC0a;
	tmp1a += *tmpB1 * tmpC0a;
	tmp2a += *tmpB2 * tmpC0a;
	tmp3a += *tmpB3 * tmpC0a;
	tmpB0 += 1;
	tmpB1 += 1;
	tmpB2 += 1;
	tmpB3 += 1;
  }
  _a[(j+0)*size+i] += tmp0a;
  _a[(j+1)*size+i] += tmp1a;
  _a[(j+2)*size+i] += tmp2a;
  _a[(j+3)*size+i] += tmp3a;
}
void cpu_kernel_jik_func_out16_inner_i4(int size, real *_a, real *_b, real *_c, int j, int i)
{
  int k;
  register real tmp0a = (real)0.0;
  register real tmp0b = (real)0.0;
  register real tmp0c = (real)0.0;
  register real tmp0d = (real)0.0;
  register real tmpC0a;
  register real tmpC0b;
  register real tmpC0c;
  register real tmpC0d;
  real *tmpB0 = &_b[(j+0)*size];
  for(k=0; k<size; k++){
	tmpC0a = _c[(k+0)*size+i];
	tmp0a += *tmpB0 * tmpC0a;
	tmpC0b = _c[(k+0)*size+i+1];
	tmp0b += *tmpB0 * tmpC0b;
	tmpC0c = _c[(k+0)*size+i+2];
	tmp0c += *tmpB0 * tmpC0c;
	tmpC0d = _c[(k+0)*size+i+3];
	tmp0d += *tmpB0 * tmpC0d;
	tmpB0 += 1;
  }
  _a[(j+0)*size+i] += tmp0a;
  _a[(j+0)*size+i+1] += tmp0b;
  _a[(j+0)*size+i+2] += tmp0c;
  _a[(j+0)*size+i+3] += tmp0d;
}
void cpu_kernel_jik_func_out16_inner_1(int size, real *_a, real *_b, real *_c, int j, int i)
{
  int k;
  register real tmp0a = (real)0.0;
  register real tmpC0a;
  real *tmpB0 = &_b[(j+0)*size];
  for(k=0; k<size; k++){
	tmpC0a = _c[(k+0)*size+i];
	tmp0a += *tmpB0 * tmpC0a;
	tmpB0 += 1;
  }
  _a[(j+0)*size+i] += tmp0a;
}
// jik、最内ループ関数化＋アンローリング、外部4X4ブロック化
void cpu_kernel_jik_func_out16_4x4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_func_out16_j4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_func_out16_i4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_func_out16_1(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_func_out16_unroll4(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size-3; j+=4){
	int i;
	for(i=0; i<size-3; i+=4){
	  cpu_kernel_jik_func_out16_4x4(size,_a,_b,_c,j,i);
	}
	if(i!=size){
	  for(i=i; i<size; i++){
		cpu_kernel_jik_func_out16_j4(size,_a,_b,_c,j,i);
	  }
	}
  }
  if(size%4!=0){
    int jbegin = size - (size/4)*4;
#pragma omp parallel for
	for(j=jbegin; j<size; j++){
	  int i;
	  for(i=0; i<size-3; i+=4){
		cpu_kernel_jik_func_out16_i4(size,_a,_b,_c,j,i);
	  }
	  if(i!=size){
		for(i=i; i<size; i++){
		  cpu_kernel_jik_func_out16_1(size,_a,_b,_c,j,i);
		}
	  }
	}
  }
}
void cpu_kernel_jik_func_out16_4x4(int size, real *_a, real *_b, real *_c, int j, int i)
{
  int k;
  register real tmp0a = (real)0.0;
  register real tmp1a = (real)0.0;
  register real tmp2a = (real)0.0;
  register real tmp3a = (real)0.0;
  register real tmp0b = (real)0.0;
  register real tmp1b = (real)0.0;
  register real tmp2b = (real)0.0;
  register real tmp3b = (real)0.0;
  register real tmp0c = (real)0.0;
  register real tmp1c = (real)0.0;
  register real tmp2c = (real)0.0;
  register real tmp3c = (real)0.0;
  register real tmp0d = (real)0.0;
  register real tmp1d = (real)0.0;
  register real tmp2d = (real)0.0;
  register real tmp3d = (real)0.0;
  register real tmpC0a;
  register real tmpC1a;
  register real tmpC2a;
  register real tmpC3a;
  register real tmpC0b;
  register real tmpC1b;
  register real tmpC2b;
  register real tmpC3b;
  register real tmpC0c;
  register real tmpC1c;
  register real tmpC2c;
  register real tmpC3c;
  register real tmpC0d;
  register real tmpC1d;
  register real tmpC2d;
  register real tmpC3d;
  real *tmpB0 = &_b[(j+0)*size];
  real *tmpB1 = &_b[(j+1)*size];
  real *tmpB2 = &_b[(j+2)*size];
  real *tmpB3 = &_b[(j+3)*size];
  for(k=0; k+3<size; k+=4){
	tmpC0a = _c[(k+0)*size+i];
	tmp0a += *tmpB0 * tmpC0a;
	tmp1a += *tmpB1 * tmpC0a;
	tmp2a += *tmpB2 * tmpC0a;
	tmp3a += *tmpB3 * tmpC0a;
	tmpC1a = _c[(k+1)*size+i];
	tmp0a += *(tmpB0+1) * tmpC1a;
	tmp1a += *(tmpB1+1) * tmpC1a;
	tmp2a += *(tmpB2+1) * tmpC1a;
	tmp3a += *(tmpB3+1) * tmpC1a;
	tmpC2a = _c[(k+2)*size+i];
	tmp0a += *(tmpB0+2) * tmpC2a;
	tmp1a += *(tmpB1+2) * tmpC2a;
	tmp2a += *(tmpB2+2) * tmpC2a;
	tmp3a += *(tmpB3+2) * tmpC2a;
	tmpC3a = _c[(k+3)*size+i];
	tmp0a += *(tmpB0+3) * tmpC3a;
	tmp1a += *(tmpB1+3) * tmpC3a;
	tmp2a += *(tmpB2+3) * tmpC3a;
	tmp3a += *(tmpB3+3) * tmpC3a;

	tmpC0b = _c[(k+0)*size+i+1];
	tmp0b += *tmpB0 * tmpC0b;
	tmp1b += *tmpB1 * tmpC0b;
	tmp2b += *tmpB2 * tmpC0b;
	tmp3b += *tmpB3 * tmpC0b;
	tmpC1b = _c[(k+1)*size+i+1];
	tmp0b += *(tmpB0+1) * tmpC1b;
	tmp1b += *(tmpB1+1) * tmpC1b;
	tmp2b += *(tmpB2+1) * tmpC1b;
	tmp3b += *(tmpB3+1) * tmpC1b;
	tmpC2b = _c[(k+2)*size+i+1];
	tmp0b += *(tmpB0+2) * tmpC2b;
	tmp1b += *(tmpB1+2) * tmpC2b;
	tmp2b += *(tmpB2+2) * tmpC2b;
	tmp3b += *(tmpB3+2) * tmpC2b;
	tmpC3b = _c[(k+3)*size+i+1];
	tmp0b += *(tmpB0+3) * tmpC3b;
	tmp1b += *(tmpB1+3) * tmpC3b;
	tmp2b += *(tmpB2+3) * tmpC3b;
	tmp3b += *(tmpB3+3) * tmpC3b;

	tmpC0c = _c[(k+0)*size+i+2];
	tmp0c += *tmpB0 * tmpC0c;
	tmp1c += *tmpB1 * tmpC0c;
	tmp2c += *tmpB2 * tmpC0c;
	tmp3c += *tmpB3 * tmpC0c;
	tmpC1c = _c[(k+1)*size+i+2];
	tmp0c += *(tmpB0+1) * tmpC1c;
	tmp1c += *(tmpB1+1) * tmpC1c;
	tmp2c += *(tmpB2+1) * tmpC1c;
	tmp3c += *(tmpB3+1) * tmpC1c;
	tmpC2c = _c[(k+2)*size+i+2];
	tmp0c += *(tmpB0+2) * tmpC2c;
	tmp1c += *(tmpB1+2) * tmpC2c;
	tmp2c += *(tmpB2+2) * tmpC2c;
	tmp3c += *(tmpB3+2) * tmpC2c;
	tmpC3c = _c[(k+3)*size+i+2];
	tmp0c += *(tmpB0+3) * tmpC3c;
	tmp1c += *(tmpB1+3) * tmpC3c;
	tmp2c += *(tmpB2+3) * tmpC3c;
	tmp3c += *(tmpB3+3) * tmpC3c;

	tmpC0d = _c[(k+0)*size+i+3];
	tmp0d += *tmpB0 * tmpC0d;
	tmp1d += *tmpB1 * tmpC0d;
	tmp2d += *tmpB2 * tmpC0d;
	tmp3d += *tmpB3 * tmpC0d;
	tmpC1d = _c[(k+1)*size+i+3];
	tmp0d += *(tmpB0+1) * tmpC1d;
	tmp1d += *(tmpB1+1) * tmpC1d;
	tmp2d += *(tmpB2+1) * tmpC1d;
	tmp3d += *(tmpB3+1) * tmpC1d;
	tmpC2d = _c[(k+2)*size+i+3];
	tmp0d += *(tmpB0+2) * tmpC2d;
	tmp1d += *(tmpB1+2) * tmpC2d;
	tmp2d += *(tmpB2+2) * tmpC2d;
	tmp3d += *(tmpB3+2) * tmpC2d;
	tmpC3d = _c[(k+3)*size+i+3];
	tmp0d += *(tmpB0+3) * tmpC3d;
	tmp1d += *(tmpB1+3) * tmpC3d;
	tmp2d += *(tmpB2+3) * tmpC3d;
	tmp3d += *(tmpB3+3) * tmpC3d;

	tmpB0 += 4;
	tmpB1 += 4;
	tmpB2 += 4;
	tmpB3 += 4;
  }
  if(k!=size){
	for(; k<size; k++){
	  tmpC0a = _c[(k+0)*size+i];
	  tmp0a += *tmpB0 * tmpC0a;
	  tmp1a += *tmpB1 * tmpC0a;
	  tmp2a += *tmpB2 * tmpC0a;
	  tmp3a += *tmpB3 * tmpC0a;

	  tmpC0b = _c[(k+0)*size+i+1];
	  tmp0b += *tmpB0 * tmpC0b;
	  tmp1b += *tmpB1 * tmpC0b;
	  tmp2b += *tmpB2 * tmpC0b;
	  tmp3b += *tmpB3 * tmpC0b;

	  tmpC0c = _c[(k+0)*size+i+2];
	  tmp0c += *tmpB0 * tmpC0c;
	  tmp1c += *tmpB1 * tmpC0c;
	  tmp2c += *tmpB2 * tmpC0c;
	  tmp3c += *tmpB3 * tmpC0c;

	  tmpC0d = _c[(k+0)*size+i+3];
	  tmp0d += *tmpB0 * tmpC0d;
	  tmp1d += *tmpB1 * tmpC0d;
	  tmp2d += *tmpB2 * tmpC0d;
	  tmp3d += *tmpB3 * tmpC0d;

	  tmpB0 += 1;
	  tmpB1 += 1;
	  tmpB2 += 1;
	  tmpB3 += 1;
	}
  }
  _a[(j+0)*size+i] += tmp0a;
  _a[(j+1)*size+i] += tmp1a;
  _a[(j+2)*size+i] += tmp2a;
  _a[(j+3)*size+i] += tmp3a;
  _a[(j+0)*size+i+1] += tmp0b;
  _a[(j+1)*size+i+1] += tmp1b;
  _a[(j+2)*size+i+1] += tmp2b;
  _a[(j+3)*size+i+1] += tmp3b;
  _a[(j+0)*size+i+2] += tmp0c;
  _a[(j+1)*size+i+2] += tmp1c;
  _a[(j+2)*size+i+2] += tmp2c;
  _a[(j+3)*size+i+2] += tmp3c;
  _a[(j+0)*size+i+3] += tmp0d;
  _a[(j+1)*size+i+3] += tmp1d;
  _a[(j+2)*size+i+3] += tmp2d;
  _a[(j+3)*size+i+3] += tmp3d;
}
void cpu_kernel_jik_func_out16_j4(int size, real *_a, real *_b, real *_c, int j, int i)
{
  int k;
  register real tmp0a = (real)0.0;
  register real tmp1a = (real)0.0;
  register real tmp2a = (real)0.0;
  register real tmp3a = (real)0.0;
  register real tmpC0a;
  register real tmpC1a;
  register real tmpC2a;
  register real tmpC3a;
  real *tmpB0 = &_b[(j+0)*size];
  real *tmpB1 = &_b[(j+1)*size];
  real *tmpB2 = &_b[(j+2)*size];
  real *tmpB3 = &_b[(j+3)*size];
  for(k=0; k+3<size; k+=4){
	tmpC0a = _c[(k+0)*size+i];
	tmp0a += *tmpB0 * tmpC0a;
	tmp1a += *tmpB1 * tmpC0a;
	tmp2a += *tmpB2 * tmpC0a;
	tmp3a += *tmpB3 * tmpC0a;
	tmpC1a = _c[(k+1)*size+i];
	tmp0a += *(tmpB0+1) * tmpC1a;
	tmp1a += *(tmpB1+1) * tmpC1a;
	tmp2a += *(tmpB2+1) * tmpC1a;
	tmp3a += *(tmpB3+1) * tmpC1a;
	tmpC2a = _c[(k+2)*size+i];
	tmp0a += *(tmpB0+2) * tmpC2a;
	tmp1a += *(tmpB1+2) * tmpC2a;
	tmp2a += *(tmpB2+2) * tmpC2a;
	tmp3a += *(tmpB3+2) * tmpC2a;
	tmpC3a = _c[(k+3)*size+i];
	tmp0a += *(tmpB0+3) * tmpC3a;
	tmp1a += *(tmpB1+3) * tmpC3a;
	tmp2a += *(tmpB2+3) * tmpC3a;
	tmp3a += *(tmpB3+3) * tmpC3a;
	tmpB0 += 4;
	tmpB1 += 4;
	tmpB2 += 4;
	tmpB3 += 4;
  }
  if(k!=size){
	for(; k<size; k++){
	  tmpC0a = _c[(k+0)*size+i];
	  tmp0a += *tmpB0 * tmpC0a;
	  tmp1a += *tmpB1 * tmpC0a;
	  tmp2a += *tmpB2 * tmpC0a;
	  tmp3a += *tmpB3 * tmpC0a;
	  tmpB0 += 1;
	  tmpB1 += 1;
	  tmpB2 += 1;
	  tmpB3 += 1;
	}
  }
  _a[(j+0)*size+i] += tmp0a;
  _a[(j+1)*size+i] += tmp1a;
  _a[(j+2)*size+i] += tmp2a;
  _a[(j+3)*size+i] += tmp3a;
}
void cpu_kernel_jik_func_out16_i4(int size, real *_a, real *_b, real *_c, int j, int i)
{
  int k;
  register real tmp0a = (real)0.0;
  register real tmp0b = (real)0.0;
  register real tmp0c = (real)0.0;
  register real tmp0d = (real)0.0;
  register real tmpC0a;
  register real tmpC1a;
  register real tmpC2a;
  register real tmpC3a;
  register real tmpC0b;
  register real tmpC1b;
  register real tmpC2b;
  register real tmpC3b;
  register real tmpC0c;
  register real tmpC1c;
  register real tmpC2c;
  register real tmpC3c;
  register real tmpC0d;
  register real tmpC1d;
  register real tmpC2d;
  register real tmpC3d;
  real *tmpB0 = &_b[(j+0)*size];
  for(k=0; k+3<size; k+=4){
	tmpC0a = _c[(k+0)*size+i];
	tmp0a += *tmpB0 * tmpC0a;
	tmpC1a = _c[(k+1)*size+i];
	tmp0a += *(tmpB0+1) * tmpC1a;
	tmpC2a = _c[(k+2)*size+i];
	tmp0a += *(tmpB0+2) * tmpC2a;
	tmpC3a = _c[(k+3)*size+i];
	tmp0a += *(tmpB0+3) * tmpC3a;

	tmpC0b = _c[(k+0)*size+i+1];
	tmp0b += *tmpB0 * tmpC0b;
	tmpC1b = _c[(k+1)*size+i+1];
	tmp0b += *(tmpB0+1) * tmpC1b;
	tmpC2b = _c[(k+2)*size+i+1];
	tmp0b += *(tmpB0+2) * tmpC2b;
	tmpC3b = _c[(k+3)*size+i+1];
	tmp0b += *(tmpB0+3) * tmpC3b;

	tmpC0c = _c[(k+0)*size+i+2];
	tmp0c += *tmpB0 * tmpC0c;
	tmpC1c = _c[(k+1)*size+i+2];
	tmp0c += *(tmpB0+1) * tmpC1c;
	tmpC2c = _c[(k+2)*size+i+2];
	tmp0c += *(tmpB0+2) * tmpC2c;
	tmpC3c = _c[(k+3)*size+i+2];
	tmp0c += *(tmpB0+3) * tmpC3c;

	tmpC0d = _c[(k+0)*size+i+3];
	tmp0d += *tmpB0 * tmpC0d;
	tmpC1d = _c[(k+1)*size+i+3];
	tmp0d += *(tmpB0+1) * tmpC1d;
	tmpC2d = _c[(k+2)*size+i+3];
	tmp0d += *(tmpB0+2) * tmpC2d;
	tmpC3d = _c[(k+3)*size+i+3];
	tmp0d += *(tmpB0+3) * tmpC3d;

	tmpB0 += 4;
  }
  if(k!=size){
	for(; k<size; k++){
	  tmpC0a = _c[(k+0)*size+i];
	  tmp0a += *tmpB0 * tmpC0a;
	  tmpC0b = _c[(k+0)*size+i+1];
	  tmp0b += *tmpB0 * tmpC0b;
	  tmpC0c = _c[(k+0)*size+i+2];
	  tmp0c += *tmpB0 * tmpC0c;
	  tmpC0d = _c[(k+0)*size+i+3];
	  tmp0d += *tmpB0 * tmpC0d;
	  tmpB0 += 1;
	}
  }
  _a[(j+0)*size+i] += tmp0a;
  _a[(j+0)*size+i+1] += tmp0b;
  _a[(j+0)*size+i+2] += tmp0c;
  _a[(j+0)*size+i+3] += tmp0d;
}
void cpu_kernel_jik_func_out16_1(int size, real *_a, real *_b, real *_c, int j, int i)
{
  int k;
  register real tmp0a = (real)0.0;
  register real tmpC0a;
  register real tmpC1a;
  register real tmpC2a;
  register real tmpC3a;
  real *tmpB0 = &_b[(j+0)*size];
  for(k=0; k+3<size; k+=4){
	tmpC0a = _c[(k+0)*size+i];
	tmp0a += *tmpB0 * tmpC0a;
	tmpC1a = _c[(k+1)*size+i];
	tmp0a += *(tmpB0+1) * tmpC1a;
	tmpC2a = _c[(k+2)*size+i];
	tmp0a += *(tmpB0+2) * tmpC2a;
	tmpC3a = _c[(k+3)*size+i];
	tmp0a += *(tmpB0+3) * tmpC3a;
	tmpB0 += 4;
  }
  if(k!=size){
	for(; k<size; k++){
	  tmpC0a = _c[(k+0)*size+i];
	  tmp0a += *tmpB0 * tmpC0a;
	  tmpB0 += 1;
	}
  }
  _a[(j+0)*size+i] += tmp0a;
}
// jik、最内ループ関数化、外側4x4ブロック化、外部制御
void cpu_kernel_4x4_inner_unroll4_j1(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_4x4_inner_unroll4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_func_out16_outcntl(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size-3; j+=4){
	int i;
	for(i=0; i<size-3; i+=4){
	  cpu_kernel_4x4_inner_unroll4(size,_a,_b,_c,j,i);
	  cpu_kernel_4x4_inner_unroll4(size,_a,_b,_c,j,i+1);
	  cpu_kernel_4x4_inner_unroll4(size,_a,_b,_c,j,i+2);
	  cpu_kernel_4x4_inner_unroll4(size,_a,_b,_c,j,i+3);
	}
	if(i!=size){
	  for(i=i; i<size; i++){
		cpu_kernel_4x4_inner_unroll4(size,_a,_b,_c,j,i);
	  }
	}
  }
  if(size%4!=0){
    int jbegin = size - (size/4)*4;
#pragma omp parallel for
	for(j=jbegin; j<size; j++){
	  int i;
	  for(i=0; i<size-3; i+=4){
		cpu_kernel_4x4_inner_unroll4_j1(size,_a,_b,_c,j,i);
		cpu_kernel_4x4_inner_unroll4_j1(size,_a,_b,_c,j,i+1);
		cpu_kernel_4x4_inner_unroll4_j1(size,_a,_b,_c,j,i+2);
		cpu_kernel_4x4_inner_unroll4_j1(size,_a,_b,_c,j,i+3);
	  }
	  if(i!=size){
		for(i=i; i<size; i++){
		  cpu_kernel_4x4_inner_unroll4_j1(size,_a,_b,_c,j,i);
		}
	  }
	}
  }
}
void cpu_kernel_4x4_inner_unroll4(int size, real *_a, real *_b, real *_c, int j, int i)
{
  int k;
  register real tmp0 = (real)0.0;
  register real tmp1 = (real)0.0;
  register real tmp2 = (real)0.0;
  register real tmp3 = (real)0.0;
  register real tmpC0;
  register real tmpC1;
  register real tmpC2;
  register real tmpC3;
  real *tmpB0 = &_b[(j+0)*size];
  real *tmpB1 = &_b[(j+1)*size];
  real *tmpB2 = &_b[(j+2)*size];
  real *tmpB3 = &_b[(j+3)*size];
  for(k=0; k+3<size; k+=4){
	tmpC0 = _c[(k+0)*size+i];
	tmp0 += *tmpB0 * tmpC0;
	tmp1 += *tmpB1 * tmpC0;
	tmp2 += *tmpB2 * tmpC0;
	tmp3 += *tmpB3 * tmpC0;
	tmpC1 = _c[(k+1)*size+i];
	tmp0 += *(tmpB0+1) * tmpC1;
	tmp1 += *(tmpB1+1) * tmpC1;
	tmp2 += *(tmpB2+1) * tmpC1;
	tmp3 += *(tmpB3+1) * tmpC1;
	tmpC2 = _c[(k+2)*size+i];
	tmp0 += *(tmpB0+2) * tmpC2;
	tmp1 += *(tmpB1+2) * tmpC2;
	tmp2 += *(tmpB2+2) * tmpC2;
	tmp3 += *(tmpB3+2) * tmpC2;
	tmpC3 = _c[(k+3)*size+i];
	tmp0 += *(tmpB0+3) * tmpC3;
	tmp1 += *(tmpB1+3) * tmpC3;
	tmp2 += *(tmpB2+3) * tmpC3;
	tmp3 += *(tmpB3+3) * tmpC3;
	tmpB0 += 4;
	tmpB1 += 4;
	tmpB2 += 4;
	tmpB3 += 4;
  }
  if(k!=size){
	for(; k<size; k++){
	  tmpC0 = _c[(k+0)*size+i];
	  tmp0 += *tmpB0 * tmpC0;
	  tmp1 += *tmpB1 * tmpC0;
	  tmp2 += *tmpB2 * tmpC0;
	  tmp3 += *tmpB3 * tmpC0;
	  tmpB0++;
	  tmpB1++;
	  tmpB2++;
	  tmpB3++;
	}
  }
  _a[(j+0)*size+i] += tmp0;
  _a[(j+1)*size+i] += tmp1;
  _a[(j+2)*size+i] += tmp2;
  _a[(j+3)*size+i] += tmp3;
}
void cpu_kernel_4x4_inner_unroll4_j1(int size, real *_a, real *_b, real *_c, int j, int i)
{
  int k;
  register real tmp0 = (real)0.0;
  register real tmpC0;
  register real tmpC1;
  register real tmpC2;
  register real tmpC3;
  real *tmpB0 = &_b[(j+0)*size];
  for(k=0; k+3<size; k+=4){
	tmpC0 = _c[(k+0)*size+i];
	tmp0 += *tmpB0 * tmpC0;
	tmpC1 = _c[(k+1)*size+i];
	tmp0 += *(tmpB0+1) * tmpC1;
	tmpC2 = _c[(k+2)*size+i];
	tmp0 += *(tmpB0+2) * tmpC2;
	tmpC3 = _c[(k+3)*size+i];
	tmp0 += *(tmpB0+3) * tmpC3;
	tmpB0 += 4;
  }
  if(k!=size){
	for(; k<size; k++){
	  tmpC0 = _c[(k+0)*size+i];
	  tmp0 += *tmpB0++ * tmpC0;
	}
  }
  _a[(j+0)*size+i] += tmp0;
}
// jik、最内ループ関数化、外側4x4ブロック化、外部制御べた書き
void cpu_kernel_jik_func_out16_flat16_inner(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_func_out16_flat16(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size-3; j+=4){
	int i;
	for(i=0; i<size-3; i+=4){
	  cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j,i);
	  cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j,i+1);
	  cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j,i+2);
	  cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j,i+3);
	  cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j+1,i);
	  cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j+1,i+1);
	  cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j+1,i+2);
	  cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j+1,i+3);
	  cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j+2,i);
	  cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j+2,i+1);
	  cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j+2,i+2);
	  cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j+2,i+3);
	  cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j+3,i);
	  cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j+3,i+1);
	  cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j+3,i+2);
	  cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j+3,i+3);
	}
	if(i!=size){
	  for(i=i; i<size; i++){
		cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j,i);
		cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j+1,i);
		cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j+2,i);
		cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j+3,i);
	  }
	}
  }
  if(size%4!=0){
    int jbegin = size - (size/4)*4;
#pragma omp parallel for
	for(j=jbegin; j<size; j++){
	  int i;
	  for(i=0; i<size-3; i+=4){
		cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j,i);
		cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j,i+1);
		cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j,i+2);
		cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j,i+3);
	  }
	  if(i!=size){
		for(i=i; i<size; i++){
		  cpu_kernel_jik_func_out16_flat16_inner(size,_a,_b,_c,j,i);
		}
	  }
	}
  }
}
void cpu_kernel_jik_func_out16_flat16_inner(int size, real *_a, real *_b, real *_c, int j, int i)
{
  int k;
  register real tmp0 = (real)0.0;
  register real tmpC;
  real *tmpB0 = &_b[(j+0)*size];
  for(k=0; k<size; k++){
	tmpC = _c[k*size+i];	tmp0 += *tmpB0++ * tmpC;
  }
  _a[(j+0)*size+i] += tmp0;
}
// jik、最内ループ関数化、外側4x4ブロック化、外部制御べた書き、関数内アンローリング
void cpu_kernel_jik_func_out16_flat16_unroll4_inner(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_func_out16_flat16_unroll4(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size-3; j+=4){
	int i;
	for(i=0; i<size-3; i+=4){
	  cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j,i);
	  cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j,i+1);
	  cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j,i+2);
	  cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j,i+3);
	  cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j+1,i);
	  cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j+1,i+1);
	  cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j+1,i+2);
	  cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j+1,i+3);
	  cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j+2,i);
	  cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j+2,i+1);
	  cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j+2,i+2);
	  cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j+2,i+3);
	  cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j+3,i);
	  cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j+3,i+1);
	  cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j+3,i+2);
	  cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j+3,i+3);
	}
	if(i!=size){
	  for(i=i; i<size; i++){
		cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j,i);
		cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j+1,i);
		cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j+2,i);
		cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j+3,i);
	  }
	}
  }
  if(size%4!=0){
    int jbegin = size - (size/4)*4;
#pragma omp parallel for
	for(j=jbegin; j<size; j++){
	  int i;
	  for(i=0; i<size-3; i+=4){
		cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j,i);
		cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j,i+1);
		cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j,i+2);
		cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j,i+3);
	  }
	  if(i!=size){
		for(i=i; i<size; i++){
		  cpu_kernel_jik_func_out16_flat16_unroll4_inner(size,_a,_b,_c,j,i);
		}
	  }
	}
  }
}
void cpu_kernel_jik_func_out16_flat16_unroll4_inner(int size, real *_a, real *_b, real *_c, int j, int i)
{
  int k;
  register real tmp0 = (real)0.0;
  register real tmpC0;
  register real tmpC1;
  register real tmpC2;
  register real tmpC3;
  real *tmpB0 = &_b[(j+0)*size];
  for(k=0; k+3<size; k+=4){
	tmpC0 = _c[(k+0)*size+i];	tmp0 += *tmpB0 * tmpC0;
	tmpC1 = _c[(k+1)*size+i];	tmp0 += *(tmpB0+1) * tmpC1;
	tmpC2 = _c[(k+2)*size+i];	tmp0 += *(tmpB0+2) * tmpC2;
	tmpC3 = _c[(k+3)*size+i];	tmp0 += *(tmpB0+3) * tmpC3;
	tmpB0 += 4;
  }
  if(k!=size){
	for(; k<size; k++){
	  tmpC0 = _c[(k+0)*size+i];	tmp0 += *tmpB0 * tmpC0;
	  tmpB0 += 1;
	}
  }
  _a[(j+0)*size+i] += tmp0;
}
// jik、最内ループ関数化、外部4x4ブロック化、計算順序微調整版
void cpu_kernel_jik_func_out16_mod_inner_4x4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_func_out16_mod_inner_j4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_func_out16_mod_inner_i4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_func_out16_mod_inner_1(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_func_out16_mod(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size-3; j+=4){
	int i;
	for(i=0; i<size-3; i+=4){
	  cpu_kernel_jik_func_out16_mod_inner_4x4(size,_a,_b,_c,j,i);
	}
	if(i!=size){
	  for(i=i; i<size; i++){
		cpu_kernel_jik_func_out16_mod_inner_j4(size,_a,_b,_c,j,i);
	  }
	}
  }
  if(size%4!=0){
    int jbegin = size - (size/4)*4;
#pragma omp parallel for
	for(j=jbegin; j<size; j++){
	  int i;
	  for(i=0; i<size-3; i+=4){
		cpu_kernel_jik_func_out16_mod_inner_i4(size,_a,_b,_c,j,i);
	  }
	  if(i!=size){
		for(i=i; i<size; i++){
		  cpu_kernel_jik_func_out16_mod_inner_1(size,_a,_b,_c,j,i);
		}
	  }
	}
  }
}
void cpu_kernel_jik_func_out16_mod_inner_4x4(int size, real *_a, real *_b, real *_c, int j, int i)
{
  int k;
  register real tmp0a = (real)0.0;
  register real tmp1a = (real)0.0;
  register real tmp2a = (real)0.0;
  register real tmp3a = (real)0.0;
  register real tmp0b = (real)0.0;
  register real tmp1b = (real)0.0;
  register real tmp2b = (real)0.0;
  register real tmp3b = (real)0.0;
  register real tmp0c = (real)0.0;
  register real tmp1c = (real)0.0;
  register real tmp2c = (real)0.0;
  register real tmp3c = (real)0.0;
  register real tmp0d = (real)0.0;
  register real tmp1d = (real)0.0;
  register real tmp2d = (real)0.0;
  register real tmp3d = (real)0.0;
  register real tmpC0a;
  register real tmpC0b;
  register real tmpC0c;
  register real tmpC0d;
  real *tmpB0 = &_b[(j+0)*size];
  real *tmpB1 = &_b[(j+1)*size];
  real *tmpB2 = &_b[(j+2)*size];
  real *tmpB3 = &_b[(j+3)*size];
  for(k=0; k<size; k++){
	tmpC0a = _c[(k+0)*size+i];
	tmpC0b = _c[(k+0)*size+i+1];
	tmpC0c = _c[(k+0)*size+i+2];
	tmpC0d = _c[(k+0)*size+i+3];

	tmp0a += *tmpB0 * tmpC0a;
	tmp0b += *tmpB0 * tmpC0b;
	tmp1a += *tmpB1 * tmpC0a;
	tmp1b += *tmpB1 * tmpC0b;
	tmp2a += *tmpB2 * tmpC0a;
	tmp2b += *tmpB2 * tmpC0b;
	tmp3a += *tmpB3 * tmpC0a;
	tmp3b += *tmpB3 * tmpC0b;

	tmp0c += *tmpB0 * tmpC0c;
	tmp0d += *tmpB0 * tmpC0d;
	tmp1c += *tmpB1 * tmpC0c;
	tmp1d += *tmpB1 * tmpC0d;
	tmp2c += *tmpB2 * tmpC0c;
	tmp2d += *tmpB2 * tmpC0d;
	tmp3c += *tmpB3 * tmpC0c;
	tmp3d += *tmpB3 * tmpC0d;

	tmpB0 += 1;
	tmpB1 += 1;
	tmpB2 += 1;
	tmpB3 += 1;
  }
  _a[(j+0)*size+i] += tmp0a;
  _a[(j+1)*size+i] += tmp1a;
  _a[(j+2)*size+i] += tmp2a;
  _a[(j+3)*size+i] += tmp3a;
  _a[(j+0)*size+i+1] += tmp0b;
  _a[(j+1)*size+i+1] += tmp1b;
  _a[(j+2)*size+i+1] += tmp2b;
  _a[(j+3)*size+i+1] += tmp3b;
  _a[(j+0)*size+i+2] += tmp0c;
  _a[(j+1)*size+i+2] += tmp1c;
  _a[(j+2)*size+i+2] += tmp2c;
  _a[(j+3)*size+i+2] += tmp3c;
  _a[(j+0)*size+i+3] += tmp0d;
  _a[(j+1)*size+i+3] += tmp1d;
  _a[(j+2)*size+i+3] += tmp2d;
  _a[(j+3)*size+i+3] += tmp3d;
}
void cpu_kernel_jik_func_out16_mod_inner_j4(int size, real *_a, real *_b, real *_c, int j, int i)
{
  int k;
  register real tmp0a = (real)0.0;
  register real tmp1a = (real)0.0;
  register real tmp2a = (real)0.0;
  register real tmp3a = (real)0.0;
  register real tmpC0a;
  real *tmpB0 = &_b[(j+0)*size];
  real *tmpB1 = &_b[(j+1)*size];
  real *tmpB2 = &_b[(j+2)*size];
  real *tmpB3 = &_b[(j+3)*size];
  for(k=0; k<size; k++){
	tmpC0a = _c[(k+0)*size+i];

	tmp0a += *tmpB0 * tmpC0a;
	tmp1a += *tmpB1 * tmpC0a;
	tmp2a += *tmpB2 * tmpC0a;
	tmp3a += *tmpB3 * tmpC0a;

	tmpB0 += 1;
	tmpB1 += 1;
	tmpB2 += 1;
	tmpB3 += 1;
  }
  _a[(j+0)*size+i] += tmp0a;
  _a[(j+1)*size+i] += tmp1a;
  _a[(j+2)*size+i] += tmp2a;
  _a[(j+3)*size+i] += tmp3a;
}
void cpu_kernel_jik_func_out16_mod_inner_i4(int size, real *_a, real *_b, real *_c, int j, int i)
{
  int k;
  register real tmp0a = (real)0.0;
  register real tmp0b = (real)0.0;
  register real tmp0c = (real)0.0;
  register real tmp0d = (real)0.0;
  register real tmpC0a;
  register real tmpC0b;
  register real tmpC0c;
  register real tmpC0d;
  real *tmpB0 = &_b[(j+0)*size];
  for(k=0; k<size; k++){
	tmpC0a = _c[(k+0)*size+i];
	tmpC0b = _c[(k+0)*size+i+1];
	tmpC0c = _c[(k+0)*size+i+2];
	tmpC0d = _c[(k+0)*size+i+3];
	tmp0a += *tmpB0 * tmpC0a;
	tmp0b += *tmpB0 * tmpC0b;
	tmp0c += *tmpB0 * tmpC0c;
	tmp0d += *tmpB0 * tmpC0d;
	tmpB0 += 1;
  }
  _a[(j+0)*size+i] += tmp0a;
  _a[(j+0)*size+i+1] += tmp0b;
  _a[(j+0)*size+i+2] += tmp0c;
  _a[(j+0)*size+i+3] += tmp0d;
}
void cpu_kernel_jik_func_out16_mod_inner_1(int size, real *_a, real *_b, real *_c, int j, int i)
{
  int k;
  register real tmp0a = (real)0.0;
  register real tmpC0a;
  real *tmpB0 = &_b[(j+0)*size];
  for(k=0; k<size; k++){
	tmpC0a = _c[(k+0)*size+i];
	tmp0a += *tmpB0 * tmpC0a;
	tmpB0 += 1;
  }
  _a[(j+0)*size+i] += tmp0a;
}

void cpu_kernel_jik_func_out16_mod2_inner_4x4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_func_out16_mod2_inner_j4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_func_out16_mod2_inner_i4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_func_out16_mod2_inner_1(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_func_out16_mod2(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size-3; j+=4){
	int i;
	for(i=0; i<size-3; i+=4){
	  cpu_kernel_jik_func_out16_mod2_inner_4x4(size,_a,_b,_c,j,i);
	}
	if(i!=size){
	  for(i=i; i<size; i++){
		cpu_kernel_jik_func_out16_mod2_inner_j4(size,_a,_b,_c,j,i);
	  }
	}
  }
  if(size%4!=0){
    int jbegin = size - (size/4)*4;
#pragma omp parallel for
	for(j=jbegin; j<size; j++){
	  int i;
	  for(i=0; i<size-3; i+=4){
		cpu_kernel_jik_func_out16_mod2_inner_i4(size,_a,_b,_c,j,i);
	  }
	  if(i!=size){
		for(i=i; i<size; i++){
		  cpu_kernel_jik_func_out16_mod2_inner_1(size,_a,_b,_c,j,i);
		}
	  }
	}
  }
}
void cpu_kernel_jik_func_out16_mod2_inner_4x4(int size, real *_a, real *_b, real *_c, int j, int i)
{
  int k;
  register real tmp0a = (real)0.0;
  register real tmp1a = (real)0.0;
  register real tmp2a = (real)0.0;
  register real tmp3a = (real)0.0;
  register real tmp0b = (real)0.0;
  register real tmp1b = (real)0.0;
  register real tmp2b = (real)0.0;
  register real tmp3b = (real)0.0;
  register real tmp0c = (real)0.0;
  register real tmp1c = (real)0.0;
  register real tmp2c = (real)0.0;
  register real tmp3c = (real)0.0;
  register real tmp0d = (real)0.0;
  register real tmp1d = (real)0.0;
  register real tmp2d = (real)0.0;
  register real tmp3d = (real)0.0;
  register real tmpC0a;
  register real tmpC0b;
  register real tmpC0c;
  register real tmpC0d;
  real *tmpB0 = &_b[(j+0)*size];
  real *tmpB1 = &_b[(j+1)*size];
  real *tmpB2 = &_b[(j+2)*size];
  real *tmpB3 = &_b[(j+3)*size];
  for(k=0; k<size; k++){
	tmpC0a = _c[(k+0)*size+i];
	tmpC0b = _c[(k+0)*size+i+1];
	tmpC0c = _c[(k+0)*size+i+2];
	tmpC0d = _c[(k+0)*size+i+3];

	tmp0a += *tmpB0 * tmpC0a;
	tmp1a += *tmpB1 * tmpC0a;
	tmp0b += *tmpB0 * tmpC0b;
	tmp1b += *tmpB1 * tmpC0b;
	tmp2a += *tmpB2 * tmpC0a;
	tmp3a += *tmpB3 * tmpC0a;
	tmp2b += *tmpB2 * tmpC0b;
	tmp3b += *tmpB3 * tmpC0b;

	tmp0c += *tmpB0 * tmpC0c;
	tmp1c += *tmpB1 * tmpC0c;
	tmp0d += *tmpB0 * tmpC0d;
	tmp1d += *tmpB1 * tmpC0d;
	tmp2c += *tmpB2 * tmpC0c;
	tmp3c += *tmpB3 * tmpC0c;
	tmp2d += *tmpB2 * tmpC0d;
	tmp3d += *tmpB3 * tmpC0d;

	tmpB0 += 1;
	tmpB1 += 1;
	tmpB2 += 1;
	tmpB3 += 1;
  }
  _a[(j+0)*size+i] += tmp0a;
  _a[(j+1)*size+i] += tmp1a;
  _a[(j+2)*size+i] += tmp2a;
  _a[(j+3)*size+i] += tmp3a;
  _a[(j+0)*size+i+1] += tmp0b;
  _a[(j+1)*size+i+1] += tmp1b;
  _a[(j+2)*size+i+1] += tmp2b;
  _a[(j+3)*size+i+1] += tmp3b;
  _a[(j+0)*size+i+2] += tmp0c;
  _a[(j+1)*size+i+2] += tmp1c;
  _a[(j+2)*size+i+2] += tmp2c;
  _a[(j+3)*size+i+2] += tmp3c;
  _a[(j+0)*size+i+3] += tmp0d;
  _a[(j+1)*size+i+3] += tmp1d;
  _a[(j+2)*size+i+3] += tmp2d;
  _a[(j+3)*size+i+3] += tmp3d;
}
void cpu_kernel_jik_func_out16_mod2_inner_j4(int size, real *_a, real *_b, real *_c, int j, int i)
{
  int k;
  register real tmp0a = (real)0.0;
  register real tmp1a = (real)0.0;
  register real tmp2a = (real)0.0;
  register real tmp3a = (real)0.0;
  register real tmpC0a;
  real *tmpB0 = &_b[(j+0)*size];
  real *tmpB1 = &_b[(j+1)*size];
  real *tmpB2 = &_b[(j+2)*size];
  real *tmpB3 = &_b[(j+3)*size];
  for(k=0; k<size; k++){
	tmpC0a = _c[(k+0)*size+i];

	tmp0a += *tmpB0 * tmpC0a;
	tmp1a += *tmpB1 * tmpC0a;
	tmp2a += *tmpB2 * tmpC0a;
	tmp3a += *tmpB3 * tmpC0a;

	tmpB0 += 1;
	tmpB1 += 1;
	tmpB2 += 1;
	tmpB3 += 1;
  }
  _a[(j+0)*size+i] += tmp0a;
  _a[(j+1)*size+i] += tmp1a;
  _a[(j+2)*size+i] += tmp2a;
  _a[(j+3)*size+i] += tmp3a;
}
void cpu_kernel_jik_func_out16_mod2_inner_i4(int size, real *_a, real *_b, real *_c, int j, int i)
{
  int k;
  register real tmp0a = (real)0.0;
  register real tmp0b = (real)0.0;
  register real tmp0c = (real)0.0;
  register real tmp0d = (real)0.0;
  register real tmpC0a;
  register real tmpC0b;
  register real tmpC0c;
  register real tmpC0d;
  real *tmpB0 = &_b[(j+0)*size];
  for(k=0; k<size; k++){
	tmpC0a = _c[(k+0)*size+i];
	tmpC0b = _c[(k+0)*size+i+1];
	tmpC0c = _c[(k+0)*size+i+2];
	tmpC0d = _c[(k+0)*size+i+3];

	tmp0a += *tmpB0 * tmpC0a;
	tmp0b += *tmpB0 * tmpC0b;
	tmp0c += *tmpB0 * tmpC0c;
	tmp0d += *tmpB0 * tmpC0d;

	tmpB0 += 1;
  }
  _a[(j+0)*size+i] += tmp0a;
  _a[(j+0)*size+i+1] += tmp0b;
  _a[(j+0)*size+i+2] += tmp0c;
  _a[(j+0)*size+i+3] += tmp0d;
}
void cpu_kernel_jik_func_out16_mod2_inner_1(int size, real *_a, real *_b, real *_c, int j, int i)
{
  int k;
  register real tmp0a = (real)0.0;
  register real tmpC0a;
  real *tmpB0 = &_b[(j+0)*size];
  for(k=0; k<size; k++){
	tmpC0a = _c[(k+0)*size+i];
	tmp0a += *tmpB0 * tmpC0a;
	tmpB0 += 1;
  }
  _a[(j+0)*size+i] += tmp0a;
}

/*
// ベクトル計算系向けの単純な実装
void cpu_kernel_jik_sse2_inner_j4(int size, real *_a, real *_b, real *_c, int j, int i)
{
  int k;
  register real tmp0a = (real)0.0;
  register real tmp1a = (real)0.0;
  register real tmp2a = (real)0.0;
  register real tmp3a = (real)0.0;
  register real tmpC0a;
  real *tmpB0 = &_b[0*size];
  real *tmpB1 = &_b[1*size];
  real *tmpB2 = &_b[2*size];
  real *tmpB3 = &_b[3*size];
  for(k=0; k<size; k++){
	tmpC0a = _c[k*size];

	tmp0a += *tmpB0 * tmpC0a;
	tmp1a += *tmpB1 * tmpC0a;
	tmp2a += *tmpB2 * tmpC0a;
	tmp3a += *tmpB3 * tmpC0a;

	tmpB0 += 1;
	tmpB1 += 1;
	tmpB2 += 1;
	tmpB3 += 1;
  }
  _a[0*size] += tmp0a;
  _a[1*size] += tmp1a;
  _a[2*size] += tmp2a;
  _a[3*size] += tmp3a;
}
void cpu_kernel_jik_sse2_inner_i4(int size, real *_a, real *_b, real *_c, int j, int i)
{
  int k;
  register real tmp0a = (real)0.0;
  register real tmp0b = (real)0.0;
  register real tmp0c = (real)0.0;
  register real tmp0d = (real)0.0;
  register real tmpC0a;
  register real tmpC0b;
  register real tmpC0c;
  register real tmpC0d;
  real *tmpB0 = &_b[0];
  for(k=0; k<size; k++){
	tmpC0a = _c[k*size];
	tmpC0b = _c[k*size+1];
	tmpC0c = _c[k*size+2];
	tmpC0d = _c[k*size+3];
	tmp0a += *tmpB0 * tmpC0a;
	tmp0b += *tmpB0 * tmpC0b;
	tmp0c += *tmpB0 * tmpC0c;
	tmp0d += *tmpB0 * tmpC0d;
	tmpB0 += 1;
  }
  _a[0] = tmp0a;
  _a[1] += tmp0b;
  _a[2] += tmp0c;
  _a[3] += tmp0d;
}
void cpu_kernel_jik_sse2_inner_1(int size, real *_a, real *_b, real *_c, int j, int i)
{
  int k;
  register real tmp0a = (real)0.0;
  register real tmpC0a;
  real *tmpB0 = &_b[0];
  for(k=0; k<size; k++){
	tmpC0a = _c[k*size];
	tmp0a += *tmpB0 * tmpC0a;
	tmpB0 += 1;
  }
  _a[0] += tmp0a;
}
*/


// ベクトル命令版
// 事実上cpu_kernel_jik_func_out16_modのSSE版
// 4x4以外はとりあえず単純版
void cpu_kernel_jik_sse_inner_4x4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_sse_inner_j4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_sse_inner_i4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_sse_inner_1(int size, real *_a, real *_b, real *_c, int j, int i);
#include "mm_base_cpu_sse_1.c"
void cpu_kernel_jik_sse(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size-3; j+=4){
	int i;
	for(i=0; i<size-3; i+=4){
	  cpu_kernel_jik_sse_inner_4x4(size,_a,_b,_c,j,i);
	}
	if(i!=size){
	  for(i=i; i<size; i++){
		cpu_kernel_jik_sse_inner_j4(size,_a,_b,_c,j,i);
	  }
	}
  }
  if(size%4!=0){
    int jbegin = size - (size/4)*4;
	//#pragma omp parallel for
	for(j=jbegin; j<size; j++){
	  int i;
	  for(i=0; i<size-3; i+=4){
		cpu_kernel_jik_sse_inner_i4(size,_a,_b,_c,j,i);
	  }
	  if(i!=size){
		for(i=i; i<size; i++){
		  cpu_kernel_jik_sse_inner_1(size,_a,_b,_c,j,i);
		}
	  }
	}
  }
}

// 外側でオフセット計算版
void cpu_kernel_jik_sse2_inner_4x4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_sse2_inner_j4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_sse2_inner_i4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_sse2_inner_1(int size, real *_a, real *_b, real *_c, int j, int i);
#include "mm_base_cpu_sse_2.c"
void cpu_kernel_jik_sse2(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size-3; j+=4){
	int i;
	for(i=0; i<size-3; i+=4){
	  cpu_kernel_jik_sse2_inner_4x4(size,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	}
	if(i!=size){
	  for(i=i; i<size; i++){
		cpu_kernel_jik_sse2_inner_j4(size,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	}
  }
  if(size%4!=0){
    int jbegin = size - (size/4)*4;
	//#pragma omp parallel for
	for(j=jbegin; j<size; j++){
	  int i;
	  for(i=0; i<size-3; i+=4){
		cpu_kernel_jik_sse2_inner_i4(size,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	  if(i!=size){
		for(i=i; i<size; i++){
		  cpu_kernel_jik_sse2_inner_1(size,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
		}
	  }
	}
  }
}
// ブロック版
void cpu_kernel_jik_sse2_blk_inner_4x4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_sse2_blk_inner_j4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_sse2_blk_inner_i4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_sse2_blk_inner_1(int size, real *_a, real *_b, real *_c, int j, int i);
#include "mm_base_cpu_sse_2_blk.c"
void cpu_kernel_jik_sse2_blk_mid(int size, int m, int n, int k, real *_a, real *_b, real *_c);
void cpu_kernel_jik_sse2_blk(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size; j+=BLK_J){
	int i,n;
	n = min(size-j,BLK_J);
	for(i=0; i<size; i+=BLK_I){
	  int k;
	  k = min(size-i,BLK_I);
	  cpu_kernel_jik_sse2_blk_mid(size, size,n,k,&_a[j*size+i],&_b[j*size],&_c[i]);
	}
  }
}
void cpu_kernel_jik_sse2_blk_mid(int size, int m, int n, int k, real *_a, real *_b, real *_c)
{
  int i, j;
  for(j=0; j+3<n; j+=4){
	for(i=0; i+3<k; i+=4){
	  cpu_kernel_jik_sse2_blk_inner_4x4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	}
	if(i!=k){
	  for(; i<k; i++){
		cpu_kernel_jik_sse2_blk_inner_j4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	}
  }
  if(j!=n){
	for(; j<n; j++){
	  for(i=0; i+3<k; i+=4){
		cpu_kernel_jik_sse2_blk_inner_i4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	  if(i!=k){
		for(; i<k; i++){
		  cpu_kernel_jik_sse2_blk_inner_1(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
		}
	  }
	}
  }
}
// ブロック版＋配列Cの不連続アクセスを一時変数で解消
// 4x4以外はとりあえず放置（解消無し）
void cpu_kernel_jik_sse2_blk_tmpc_inner_4x4(int size, real *_a, real *_b, real *_c, int j, int i);
//void cpu_kernel_jik_sse2_blk_tmpc_inner_j4(int size, real *_a, real *_b, real *_c, int j, int i);
//void cpu_kernel_jik_sse2_blk_tmpc_inner_i4(int size, real *_a, real *_b, real *_c, int j, int i);
//void cpu_kernel_jik_sse2_blk_tmpc_inner_1(int size, real *_a, real *_b, real *_c, int j, int i);
#include "mm_base_cpu_sse_2_blk_tmpc.c"
void cpu_kernel_jik_sse2_blk_tmpc_mid(int size, int m, int n, int k, real *_a, real *_b, real *_c);
void cpu_kernel_jik_sse2_blk_tmpc(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size; j+=BLK_J){
	int i, n;
	n = min(size-j,BLK_J);
	for(i=0; i<size; i+=BLK_I){
	  int k;
	  k = min(size-i,BLK_I);
	  cpu_kernel_jik_sse2_blk_tmpc_mid(size, size,n,k,&_a[j*size+i],&_b[j*size],&_c[i]);
	}
  }
}
void cpu_kernel_jik_sse2_blk_tmpc_mid(int size, int m, int n, int k, real *_a, real *_b, real *_c)
{
  int i, j;
  double tmpc[m*k];
  for(j=0; j+3<n; j+=4){
	for(i=0; i+3<k; i+=4){
	  int x;
	  for(x=0;x<size;x++){
		tmpc[x*4+0] = _c[i+x*size+0];
		tmpc[x*4+1] = _c[i+x*size+1];
		tmpc[x*4+2] = _c[i+x*size+2];
		tmpc[x*4+3] = _c[i+x*size+3];
	  }
	  cpu_kernel_jik_sse2_blk_tmpc_inner_4x4(m,&_a[j*size+i],&_b[j*size],&tmpc[0],j,i);
	}
	if(i!=k){
	  for(; i<k; i++){
		cpu_kernel_jik_sse2_blk_inner_j4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	}
  }
  if(j!=n){
	for(; j<n; j++){
	  for(i=0; i+3<k; i+=4){
		cpu_kernel_jik_sse2_blk_inner_i4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	  if(i!=k){
		for(; i<k; i++){
		  cpu_kernel_jik_sse2_blk_inner_1(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
		}
	  }
	}
  }
}

// ブロック版＋配列Cの不連続アクセスを一時変数で解消、格納処理をまとめる
// 4x4以外はとりあえず放置（解消無し）
void cpu_kernel_jik_sse2_blk_tmpc2_inner_4x4(int size, real *_a, real *_b, real *_c, int j, int i);
//void cpu_kernel_jik_sse2_blk_tmpc2_inner_j4(int size, real *_a, real *_b, real *_c, int j, int i);
//void cpu_kernel_jik_sse2_blk_tmpc2_inner_i4(int size, real *_a, real *_b, real *_c, int j, int i);
//void cpu_kernel_jik_sse2_blk_tmpc2_inner_1(int size, real *_a, real *_b, real *_c, int j, int i);
#include "mm_base_cpu_sse_2_blk_tmpc2.c"
void cpu_kernel_jik_sse2_blk_tmpc2_mid(int size, int m, int n, int k, real *_a, real *_b, real *_c);
void cpu_kernel_jik_sse2_blk_tmpc2(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size; j+=BLK_J){
	int i, n;
	n = min(size-j,BLK_J);
	for(i=0; i<size; i+=BLK_I){
	  int k;
	  k = min(size-i,BLK_I);
	  cpu_kernel_jik_sse2_blk_tmpc2_mid(size, size,n,k,&_a[j*size+i],&_b[j*size],&_c[i]);
	}
  }
}
void cpu_kernel_jik_sse2_blk_tmpc2_mid(int size, int m, int n, int k, real *_a, real *_b, real *_c)
{
  int i, j;
  double tmpc[m*k];
  for(j=0; j+3<n; j+=4){
	for(i=0; i+3<k; i+=4){
	  int x;
	  if(j==0){
		for(x=0;x<size;x++){
		  tmpc[i*m+x*4+0] = _c[i+x*size+0];
		  tmpc[i*m+x*4+1] = _c[i+x*size+1];
		  tmpc[i*m+x*4+2] = _c[i+x*size+2];
		  tmpc[i*m+x*4+3] = _c[i+x*size+3];
		}
	  }
	  cpu_kernel_jik_sse2_blk_tmpc2_inner_4x4(m,&_a[j*size+i],&_b[j*size],&tmpc[m*i],j,i);
	}
	if(i!=k){
	  for(; i<k; i++){
		cpu_kernel_jik_sse2_blk_inner_j4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	}
  }
  if(j!=n){
	for(; j<n; j++){
	  for(i=0; i+3<k; i+=4){
		cpu_kernel_jik_sse2_blk_inner_i4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	  if(i!=k){
		for(; i<k; i++){
		  cpu_kernel_jik_sse2_blk_inner_1(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
		}
	  }
	}
  }
}

// ブロック版＋配列Cの不連続アクセスを一時変数で解消、格納処理をまとめる
// Bも一時変数に格納
// 4x4以外はとりあえず放置（解消無し）
void cpu_kernel_jik_sse2_blk_tmpc2_tmpb_inner_4x4(int size, real *_a, real *_b, real *_c, int j, int i);
//void cpu_kernel_jik_sse2_blk_tmpc2_tmpb_inner_j4(int size, real *_a, real *_b, real *_c, int j, int i);
//void cpu_kernel_jik_sse2_blk_tmpc2_tmpb_inner_i4(int size, real *_a, real *_b, real *_c, int j, int i);
//void cpu_kernel_jik_sse2_blk_tmpc2_tmpb_inner_1(int size, real *_a, real *_b, real *_c, int j, int i);
#include "mm_base_cpu_sse_2_blk_tmpc2_tmpb.c"
void cpu_kernel_jik_sse2_blk_tmpc2_tmpb_mid(int size, int m, int n, int k, real *_a, real *_b, real *_c);
void cpu_kernel_jik_sse2_blk_tmpc2_tmpb(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size; j+=BLK_J){
	int i, n;
	n = min(size-j,BLK_J);
	for(i=0; i<size; i+=BLK_I){
	  int k;
	  k = min(size-i,BLK_I);
	  cpu_kernel_jik_sse2_blk_tmpc2_tmpb_mid(size, size,n,k,&_a[j*size+i],&_b[j*size],&_c[i]);
	}
  }
}
void cpu_kernel_jik_sse2_blk_tmpc2_tmpb_mid(int size, int m, int n, int k, real *_a, real *_b, real *_c)
{
  int i, j, x;
  double tmpc[m*k], tmpb[m*n];
  for(j=0; j+3<n; j+=4){
	double *tmpB0 = &_b[(j+0)*size];
	double *tmpB1 = &_b[(j+1)*size];
	double *tmpB2 = &_b[(j+2)*size];
	double *tmpB3 = &_b[(j+3)*size];
	int y=0;
	for(x=0;x<size;x+=1){
	  tmpb[y++] = *tmpB0++;
	  tmpb[y++] = *tmpB1++;
	  tmpb[y++] = *tmpB2++;
	  tmpb[y++] = *tmpB3++;
	}
	for(i=0; i+3<k; i+=4){
	  if(j==0){
		for(x=0;x<size;x++){
		  tmpc[i*m+x*4+0] = _c[i+x*size+0];
		  tmpc[i*m+x*4+1] = _c[i+x*size+1];
		  tmpc[i*m+x*4+2] = _c[i+x*size+2];
		  tmpc[i*m+x*4+3] = _c[i+x*size+3];
		}
	  }
	  cpu_kernel_jik_sse2_blk_tmpc2_tmpb_inner_4x4(m,&_a[j*size+i],&tmpb[0],&tmpc[m*i],j,i);
	}
	if(i!=k){
	  for(; i<k; i++){
		cpu_kernel_jik_sse2_blk_inner_j4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	}
  }
  if(j!=n){
	for(; j<n; j++){
	  for(i=0; i+3<k; i+=4){
		cpu_kernel_jik_sse2_blk_inner_i4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	  if(i!=k){
		for(; i<k; i++){
		  cpu_kernel_jik_sse2_blk_inner_1(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
		}
	  }
	}
  }
}


// ブロック版＋配列Cの不連続アクセスを一時変数で解消、格納処理をまとめる
// Bも一時変数に格納、格納を最初だけに制限（並列実行時に注意）
// 4x4以外はとりあえず放置（解消無し）
void cpu_kernel_jik_sse2_blk_tmpc2_tmpb2_inner_4x4(int size, real *_a, real *_b, real *_c, int j, int i);
//void cpu_kernel_jik_sse2_blk_tmpc2_tmpb2_inner_j4(int size, real *_a, real *_b, real *_c, int j, int i);
//void cpu_kernel_jik_sse2_blk_tmpc2_tmpb2_inner_i4(int size, real *_a, real *_b, real *_c, int j, int i);
//void cpu_kernel_jik_sse2_blk_tmpc2_tmpb2_inner_1(int size, real *_a, real *_b, real *_c, int j, int i);
#include "mm_base_cpu_sse_2_blk_tmpc2_tmpb2.c"
void cpu_kernel_jik_sse2_blk_tmpc2_tmpb2_mid(int size, int m, int n, int k, real *_a, real *_b, real *_c, int first);
void cpu_kernel_jik_sse2_blk_tmpc2_tmpb2(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size; j+=BLK_J){
	int i, n;
	n = min(size-j,BLK_J);
	for(i=0; i<size; i+=BLK_I){
	  int k;
	  k = min(size-i,BLK_I);
	  cpu_kernel_jik_sse2_blk_tmpc2_tmpb2_mid(size, size,n,k,&_a[j*size+i],&_b[j*size],&_c[i],i==0);
	}
  }
}
void cpu_kernel_jik_sse2_blk_tmpc2_tmpb2_mid(int size, int m, int n, int k, real *_a, real *_b, real *_c, int first)
{
  int i, j, x;
  double tmpc[m*k];
  static double tmpb[BSIZE]; // caution for parallel programs
  for(j=0; j+3<n; j+=4){
	if(first==1){
	  double *tmpB0 = &_b[(j+0)*size];
	  double *tmpB1 = &_b[(j+1)*size];
	  double *tmpB2 = &_b[(j+2)*size];
	  double *tmpB3 = &_b[(j+3)*size];
	  int y=0;
	  for(x=0;x<size;x+=1){
		tmpb[y++] = *tmpB0++;
		tmpb[y++] = *tmpB1++;
		tmpb[y++] = *tmpB2++;
		tmpb[y++] = *tmpB3++;
	  }
	}
	for(i=0; i+3<k; i+=4){
	  if(j==0){
		double *_cc;
		double *_tmpc = &tmpc[i*m];
		for(x=0;x<size;x++){
		  // いずれも有意な差は生じない模様
		  /*
		  tmpc[i*m+x*4+0] = _c[i+x*size+0];
		  tmpc[i*m+x*4+1] = _c[i+x*size+1];
		  tmpc[i*m+x*4+2] = _c[i+x*size+2];
		  tmpc[i*m+x*4+3] = _c[i+x*size+3];
		  */
		  /*
		  _cc = &_c[i+x*size];
		  tmpc[i*m+x*4+0] = *_cc++;
		  tmpc[i*m+x*4+1] = *_cc++;
		  tmpc[i*m+x*4+2] = *_cc++;
		  tmpc[i*m+x*4+3] = *_cc++;
		  */
		  _cc = &_c[i+x*size];
		  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;
		}
	  }
	  cpu_kernel_jik_sse2_blk_tmpc2_tmpb2_inner_4x4(m,&_a[j*size+i],&tmpb[0],&tmpc[m*i],j,i);
	}
	if(i!=k){
	  for(; i<k; i++){
		cpu_kernel_jik_sse2_blk_inner_j4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	}
  }
  if(j!=n){
	for(; j<n; j++){
	  for(i=0; i+3<k; i+=4){
		cpu_kernel_jik_sse2_blk_inner_i4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	  if(i!=k){
		for(; i<k; i++){
		  cpu_kernel_jik_sse2_blk_inner_1(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
		}
	  }
	}
  }
}

// 34に対して最内アンローリングテスト（どちらが高速？）
void cpu_kernel_35_inner(int size, real *_a, real *_b, real *_c, int j, int i)
{
#ifdef _USE_SSE
  int k;
  v2df_t v_0a0b, v_1a1b, v_2a2b, v_3a3b;
  v2df_t v_0c0d, v_1c1d, v_2c2d, v_3c3d;
  v2df_t v_b0_1, v_b1_1, v_b2_1, v_b3_1;
  v2df_t v_b0_2, v_b1_2, v_b2_2, v_b3_2;
  v2df_t v_c0a0b_1, v_c0c0d_1;
  v2df_t v_c0a0b_2, v_c0c0d_2;
  v_0a0b.v = _mm_setzero_pd();
  v_1a1b.v = _mm_setzero_pd();
  v_2a2b.v = _mm_setzero_pd();
  v_3a3b.v = _mm_setzero_pd();
  v_0c0d.v = _mm_setzero_pd();
  v_1c1d.v = _mm_setzero_pd();
  v_2c2d.v = _mm_setzero_pd();
  v_3c3d.v = _mm_setzero_pd();
  for(k=0; k<size; k+=2){
	v_c0a0b_1.v = _mm_load_pd((double*)_c); _c+=2;
	v_c0c0d_1.v = _mm_load_pd((double*)_c); _c+=2;
	v_b0_1.v = _mm_loaddup_pd((double*)_b++);
	v_b1_1.v = _mm_loaddup_pd((double*)_b++);
	v_b2_1.v = _mm_loaddup_pd((double*)_b++);
	v_b3_1.v = _mm_loaddup_pd((double*)_b++);
#ifdef _USE_MM_INST
	v_0a0b.v = _mm_add_pd(v_0a0b.v, _mm_mul_pd(v_b0_1.v, v_c0a0b_1.v));
	v_1a1b.v = _mm_add_pd(v_1a1b.v, _mm_mul_pd(v_b1_1.v, v_c0a0b_1.v));
	v_2a2b.v = _mm_add_pd(v_2a2b.v, _mm_mul_pd(v_b2_1.v, v_c0a0b_1.v));
	v_3a3b.v = _mm_add_pd(v_3a3b.v, _mm_mul_pd(v_b3_1.v, v_c0a0b_1.v));
	v_0c0d.v = _mm_add_pd(v_0c0d.v, _mm_mul_pd(v_b0_1.v, v_c0c0d_1.v));
	v_1c1d.v = _mm_add_pd(v_1c1d.v, _mm_mul_pd(v_b1_1.v, v_c0c0d_1.v));
	v_2c2d.v = _mm_add_pd(v_2c2d.v, _mm_mul_pd(v_b2_1.v, v_c0c0d_1.v));
	v_3c3d.v = _mm_add_pd(v_3c3d.v, _mm_mul_pd(v_b3_1.v, v_c0c0d_1.v));
#else
	v_0a0b.v += v_b0_1.v * v_c0a0b_1.v;	v_1a1b.v += v_b1_1.v * v_c0a0b_1.v;
	v_2a2b.v += v_b2_1.v * v_c0a0b_1.v;	v_3a3b.v += v_b3_1.v * v_c0a0b_1.v;
	v_0c0d.v += v_b0_1.v * v_c0c0d_1.v;	v_1c1d.v += v_b1_1.v * v_c0c0d_1.v;
	v_2c2d.v += v_b2_1.v * v_c0c0d_1.v;	v_3c3d.v += v_b3_1.v * v_c0c0d_1.v;
#endif
	v_c0a0b_2.v = _mm_load_pd((double*)_c); _c+=2;
	v_c0c0d_2.v = _mm_load_pd((double*)_c); _c+=2;
	v_b0_2.v = _mm_loaddup_pd((double*)_b++);
	v_b1_2.v = _mm_loaddup_pd((double*)_b++);
	v_b2_2.v = _mm_loaddup_pd((double*)_b++);
	v_b3_2.v = _mm_loaddup_pd((double*)_b++);
#ifdef _USE_MM_INST
	v_0a0b.v = _mm_add_pd(v_0a0b.v, _mm_mul_pd(v_b0_2.v, v_c0a0b_2.v));
	v_1a1b.v = _mm_add_pd(v_1a1b.v, _mm_mul_pd(v_b1_2.v, v_c0a0b_2.v));
	v_2a2b.v = _mm_add_pd(v_2a2b.v, _mm_mul_pd(v_b2_2.v, v_c0a0b_2.v));
	v_3a3b.v = _mm_add_pd(v_3a3b.v, _mm_mul_pd(v_b3_2.v, v_c0a0b_2.v));
	v_0c0d.v = _mm_add_pd(v_0c0d.v, _mm_mul_pd(v_b0_2.v, v_c0c0d_2.v));
	v_1c1d.v = _mm_add_pd(v_1c1d.v, _mm_mul_pd(v_b1_2.v, v_c0c0d_2.v));
	v_2c2d.v = _mm_add_pd(v_2c2d.v, _mm_mul_pd(v_b2_2.v, v_c0c0d_2.v));
	v_3c3d.v = _mm_add_pd(v_3c3d.v, _mm_mul_pd(v_b3_2.v, v_c0c0d_2.v));
#else
	v_0a0b.v += v_b0_2.v * v_c0a0b_2.v;	v_1a1b.v += v_b1_2.v * v_c0a0b_2.v;
	v_2a2b.v += v_b2_2.v * v_c0a0b_2.v;	v_3a3b.v += v_b3_2.v * v_c0a0b_2.v;
	v_0c0d.v += v_b0_2.v * v_c0c0d_2.v;	v_1c1d.v += v_b1_2.v * v_c0c0d_2.v;
	v_2c2d.v += v_b2_2.v * v_c0c0d_2.v;	v_3c3d.v += v_b3_2.v * v_c0c0d_2.v;
#endif
  }
  _a[0*size] += v_0a0b.d[0];
  _a[1*size] += v_1a1b.d[0];
  _a[2*size] += v_2a2b.d[0];
  _a[3*size] += v_3a3b.d[0];
  _a[0*size+1] += v_0a0b.d[1];
  _a[1*size+1] += v_1a1b.d[1];
  _a[2*size+1] += v_2a2b.d[1];
  _a[3*size+1] += v_3a3b.d[1];
  _a[0*size+2] += v_0c0d.d[0];
  _a[1*size+2] += v_1c1d.d[0];
  _a[2*size+2] += v_2c2d.d[0];
  _a[3*size+2] += v_3c3d.d[0];
  _a[0*size+3] += v_0c0d.d[1];
  _a[1*size+3] += v_1c1d.d[1];
  _a[2*size+3] += v_2c2d.d[1];
  _a[3*size+3] += v_3c3d.d[1];
#endif
}
void cpu_kernel_35_blk(int size, int m, int n, int k, real *_a, real *_b, real *_c, int first)
{
  int i, j, x;
  double tmpc[m*k];
  static double tmpb[BSIZE]; // caution for parallel programs
  for(j=0; j<n; j+=4){
	if(first==1){
	  double *tmpB0 = &_b[(j+0)*size];
	  double *tmpB1 = &_b[(j+1)*size];
	  double *tmpB2 = &_b[(j+2)*size];
	  double *tmpB3 = &_b[(j+3)*size];
	  int y=0;
	  for(x=0;x<size;x+=1){
		tmpb[y++] = *tmpB0++;
		tmpb[y++] = *tmpB1++;
		tmpb[y++] = *tmpB2++;
		tmpb[y++] = *tmpB3++;
	  }
	}
	for(i=0; i<k; i+=4){
	  if(j==0){
		double *_cc;
		double *_tmpc = &tmpc[i*m];
		for(x=0;x<size;x++){
		  // 有意な差は生じない模様
		  /*
		  tmpc[i*m+x*4+0] = _c[i+x*size+0];
		  tmpc[i*m+x*4+1] = _c[i+x*size+1];
		  tmpc[i*m+x*4+2] = _c[i+x*size+2];
		  tmpc[i*m+x*4+3] = _c[i+x*size+3];
		  */
		  /*
		  _cc = &_c[i+x*size];
		  tmpc[i*m+x*4+0] = *_cc++;
		  tmpc[i*m+x*4+1] = *_cc++;
		  tmpc[i*m+x*4+2] = *_cc++;
		  tmpc[i*m+x*4+3] = *_cc++;
		  */
		  _cc = &_c[i+x*size];
		  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;
		}
	  }
	  cpu_kernel_35_inner(m,&_a[j*size+i],&tmpb[0],&tmpc[m*i],j,i);
	}
  }
}
void cpu_kernel_35(int kernel, int size, real *_a, real *_b, real *_c)
{
  int i, j;
  int n, k;
  for(j=0; j<size; j+=BLK_J){
	n = min(size-j,BLK_J);
	for(i=0; i<size; i+=BLK_I){
	  k = min(size-i,BLK_I);
	  cpu_kernel_35_blk(size, size,n,k,&_a[j*size+i],&_b[j*size],&_c[i],i==0);
	}
  }
}
// 32の_c別版
void cpu_kernel_36_inner(int size, real *_a, real *_b, real *_c, int j, int i)
{
#ifdef _USE_SSE
  int k;
  real *tmpB0 = &_b[0*size];
  real *tmpB1 = &_b[1*size];
  real *tmpB2 = &_b[2*size];
  real *tmpB3 = &_b[3*size];
  v2df_t v_0a0b, v_1a1b, v_2a2b, v_3a3b;
  v2df_t v_0c0d, v_1c1d, v_2c2d, v_3c3d;
  v2df_t v_b0, v_b1, v_b2, v_b3;
  v2df_t v_c0a0b, v_c0c0d;
  v_0a0b.v = _mm_setzero_pd();
  v_1a1b.v = _mm_setzero_pd();
  v_2a2b.v = _mm_setzero_pd();
  v_3a3b.v = _mm_setzero_pd();
  v_0c0d.v = _mm_setzero_pd();
  v_1c1d.v = _mm_setzero_pd();
  v_2c2d.v = _mm_setzero_pd();
  v_3c3d.v = _mm_setzero_pd();
  for(k=0; k<size; k++){
	// 資料では32ではなくこちらを使っていたが、逆に遅くなってしまう気がする
	v_c0a0b.v = _mm_load_pd((double*)_c);
	v_c0c0d.v = _mm_load_pd((double*)(_c+2));
	_c+=4;
	// 32版
	//v_c0a0b.v = _mm_load_pd((double*)&_c[k*4]);
	//v_c0c0d.v = _mm_load_pd((double*)&_c[k*4+2]);
	v_b0.v = _mm_loaddup_pd((double*)tmpB0++);
	v_b1.v = _mm_loaddup_pd((double*)tmpB1++);
	v_b2.v = _mm_loaddup_pd((double*)tmpB2++);
	v_b3.v = _mm_loaddup_pd((double*)tmpB3++);
#ifdef _USE_MM_INST
	v_0a0b.v = _mm_add_pd(v_0a0b.v, _mm_mul_pd(v_b0.v, v_c0a0b.v));
	v_1a1b.v = _mm_add_pd(v_1a1b.v, _mm_mul_pd(v_b1.v, v_c0a0b.v));
	v_2a2b.v = _mm_add_pd(v_2a2b.v, _mm_mul_pd(v_b2.v, v_c0a0b.v));
	v_3a3b.v = _mm_add_pd(v_3a3b.v, _mm_mul_pd(v_b3.v, v_c0a0b.v));
	v_0c0d.v = _mm_add_pd(v_0c0d.v, _mm_mul_pd(v_b0.v, v_c0c0d.v));
	v_1c1d.v = _mm_add_pd(v_1c1d.v, _mm_mul_pd(v_b1.v, v_c0c0d.v));
	v_2c2d.v = _mm_add_pd(v_2c2d.v, _mm_mul_pd(v_b2.v, v_c0c0d.v));
	v_3c3d.v = _mm_add_pd(v_3c3d.v, _mm_mul_pd(v_b3.v, v_c0c0d.v));
#else
	v_0a0b.v += v_b0.v * v_c0a0b.v;
	v_1a1b.v += v_b1.v * v_c0a0b.v;
	v_2a2b.v += v_b2.v * v_c0a0b.v;
	v_3a3b.v += v_b3.v * v_c0a0b.v;
	v_0c0d.v += v_b0.v * v_c0c0d.v;
	v_1c1d.v += v_b1.v * v_c0c0d.v;
	v_2c2d.v += v_b2.v * v_c0c0d.v;
	v_3c3d.v += v_b3.v * v_c0c0d.v;
#endif
  }
  _a[0*size] += v_0a0b.d[0];
  _a[1*size] += v_1a1b.d[0];
  _a[2*size] += v_2a2b.d[0];
  _a[3*size] += v_3a3b.d[0];
  _a[0*size+1] += v_0a0b.d[1];
  _a[1*size+1] += v_1a1b.d[1];
  _a[2*size+1] += v_2a2b.d[1];
  _a[3*size+1] += v_3a3b.d[1];
  _a[0*size+2] += v_0c0d.d[0];
  _a[1*size+2] += v_1c1d.d[0];
  _a[2*size+2] += v_2c2d.d[0];
  _a[3*size+2] += v_3c3d.d[0];
  _a[0*size+3] += v_0c0d.d[1];
  _a[1*size+3] += v_1c1d.d[1];
  _a[2*size+3] += v_2c2d.d[1];
  _a[3*size+3] += v_3c3d.d[1];
#endif
}
void cpu_kernel_36_blk(int size, int m, int n, int k, real *_a, real *_b, real *_c)
{
  int i, j;
  double tmpc[m*k];
  for(j=0; j<n; j+=4){
	for(i=0; i<k; i+=4){
	  int x;
	  if(j==0){
		for(x=0;x<size;x++){
		  tmpc[i*m+x*4+0] = _c[i+x*size+0];
		  tmpc[i*m+x*4+1] = _c[i+x*size+1];
		  tmpc[i*m+x*4+2] = _c[i+x*size+2];
		  tmpc[i*m+x*4+3] = _c[i+x*size+3];
		}
	  }
	  cpu_kernel_36_inner(m,&_a[j*size+i],&_b[j*size],&tmpc[m*i],j,i);
	}
  }
}
void cpu_kernel_36(int kernel, int size, real *_a, real *_b, real *_c)
{
  int i, j;
  int n, k;
  for(j=0; j<size; j+=BLK_J){
	n = min(size-j,BLK_J);
	for(i=0; i<size; i+=BLK_I){
	  k = min(size-i,BLK_I);
	  cpu_kernel_36_blk(size, size,n,k,&_a[j*size+i],&_b[j*size],&_c[i]);
	}
  }
}
// 26を8単位に変更（未遂）
void cpu_kernel_37_inner(int size, real *_a, real *_b, real *_c, int j, int i)
{
  int k;
  register real tmp0a = (real)0.0;
  register real tmp1a = (real)0.0;
  register real tmp2a = (real)0.0;
  register real tmp3a = (real)0.0;
  register real tmp0b = (real)0.0;
  register real tmp1b = (real)0.0;
  register real tmp2b = (real)0.0;
  register real tmp3b = (real)0.0;
  register real tmp0c = (real)0.0;
  register real tmp1c = (real)0.0;
  register real tmp2c = (real)0.0;
  register real tmp3c = (real)0.0;
  register real tmp0d = (real)0.0;
  register real tmp1d = (real)0.0;
  register real tmp2d = (real)0.0;
  register real tmp3d = (real)0.0;
  register real tmpC0a;
  register real tmpC0b;
  register real tmpC0c;
  register real tmpC0d;
  real *tmpB0 = &_b[(j+0)*size];
  real *tmpB1 = &_b[(j+1)*size];
  real *tmpB2 = &_b[(j+2)*size];
  real *tmpB3 = &_b[(j+3)*size];
  for(k=0; k<size; k++){
	tmpC0a = _c[(k+0)*size+i];
	tmpC0b = _c[(k+0)*size+i+1];
	tmpC0c = _c[(k+0)*size+i+2];
	tmpC0d = _c[(k+0)*size+i+3];

	tmp0a += *tmpB0 * tmpC0a;
	tmp0b += *tmpB0 * tmpC0b;
	tmp1a += *tmpB1 * tmpC0a;
	tmp1b += *tmpB1 * tmpC0b;
	tmp2a += *tmpB2 * tmpC0a;
	tmp2b += *tmpB2 * tmpC0b;
	tmp3a += *tmpB3 * tmpC0a;
	tmp3b += *tmpB3 * tmpC0b;

	tmp0c += *tmpB0 * tmpC0c;
	tmp0d += *tmpB0 * tmpC0d;
	tmp1c += *tmpB1 * tmpC0c;
	tmp1d += *tmpB1 * tmpC0d;
	tmp2c += *tmpB2 * tmpC0c;
	tmp2d += *tmpB2 * tmpC0d;
	tmp3c += *tmpB3 * tmpC0c;
	tmp3d += *tmpB3 * tmpC0d;

	tmpB0 += 1;
	tmpB1 += 1;
	tmpB2 += 1;
	tmpB3 += 1;
  }
  _a[(j+0)*size+i] += tmp0a;
  _a[(j+1)*size+i] += tmp1a;
  _a[(j+2)*size+i] += tmp2a;
  _a[(j+3)*size+i] += tmp3a;
  _a[(j+0)*size+i+1] += tmp0b;
  _a[(j+1)*size+i+1] += tmp1b;
  _a[(j+2)*size+i+1] += tmp2b;
  _a[(j+3)*size+i+1] += tmp3b;
  _a[(j+0)*size+i+2] += tmp0c;
  _a[(j+1)*size+i+2] += tmp1c;
  _a[(j+2)*size+i+2] += tmp2c;
  _a[(j+3)*size+i+2] += tmp3c;
  _a[(j+0)*size+i+3] += tmp0d;
  _a[(j+1)*size+i+3] += tmp1d;
  _a[(j+2)*size+i+3] += tmp2d;
  _a[(j+3)*size+i+3] += tmp3d;
}
void cpu_kernel_37(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size; j+=8){
	int i;
	for(i=0; i<size; i+=8){
	  cpu_kernel_37_inner(size,_a,_b,_c,j,i);
	}
  }
}

// AVX
// 4x4以外はとりあえず適当
void cpu_kernel_jik_avx_blk(int size, int m, int n, int k, real *_a, real *_b, real *_c, int first);
#include "mm_base_cpu_avx.c"
void cpu_kernel_jik_avx_4x4
(MYCONST int size, real * MYRESTRICT _a, MYCONST real * MYRESTRICT _b, MYCONST real * MYRESTRICT _c,
 MYCONST int j, MYCONST int i);
//void cpu_kernel_jik_avx_4x4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_avx(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size; j+=BLK_J){
	int i, n;
	n = min(size-j,BLK_J);
	for(i=0; i<size; i+=BLK_I){
	  int k;
	  k = min(size-i,BLK_I);
	  cpu_kernel_jik_avx_blk(size, size,n,k,&_a[j*size+i],&_b[j*size],&_c[i],i==0);
	}
  }
}
void cpu_kernel_jik_avx_blk(int size, int m, int n, int k, real *_a, real *_b, real *_c, int first)
{
  int i, j, x;
  double tmpc[m*k], tmpb[m*n];
  for(j=0; j+3<n; j+=4){
	double *tmpB0 = &_b[(j+0)*size];
	double *tmpB1 = &_b[(j+1)*size];
	double *tmpB2 = &_b[(j+2)*size];
	double *tmpB3 = &_b[(j+3)*size];
	int y=0;
	for(x=0;x<size;x+=1){
	  tmpb[y++] = *tmpB0++;
	  tmpb[y++] = *tmpB1++;
	  tmpb[y++] = *tmpB2++;
	  tmpb[y++] = *tmpB3++;
	}
	for(i=0; i+3<k; i+=4){
	  if(j==0){
		double *_cc;
		double *_tmpc = &tmpc[i*m];
		for(x=0;x<size;x++){
		  _cc = &_c[i+x*size];
		  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;
		}
	  }
	  cpu_kernel_jik_avx_4x4(m,&_a[j*size+i],&tmpb[0],&tmpc[m*i],j,i);
	}
	if(i!=k){
	  for(; i<k; i++){
		cpu_kernel_jik_sse2_blk_inner_j4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	}
  }
  if(j!=n){
	for(; j<n; j++){
	  for(i=0; i+3<k; i+=4){
		cpu_kernel_jik_sse2_blk_inner_i4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	  if(i!=k){
		for(; i<k; i++){
		  cpu_kernel_jik_sse2_blk_inner_1(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
		}
	  }
	}
  }
}
// Bの一時変数static化（OpenMP非対応）
void cpu_kernel_jik_avx_tmpb2_blk(int size, int m, int n, int k, real *_a, real *_b, real *_c, int first);
#include "mm_base_cpu_avx.c"
void cpu_kernel_jik_avx_tmpb2(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size; j+=BLK_J){
	int i, n;
	n = min(size-j,BLK_J);
	for(i=0; i<size; i+=BLK_I){
	  int k;
	  k = min(size-i,BLK_I);
	  cpu_kernel_jik_avx_tmpb2_blk(size, size,n,k,&_a[j*size+i],&_b[j*size],&_c[i],i==0);
	}
  }
}
void cpu_kernel_jik_avx_tmpb2_blk(int size, int m, int n, int k, real *_a, real *_b, real *_c, int first)
{
  int i, j, x;
  double tmpc[m*k];
  static double tmpb[BSIZE]; // caution for parallel programs
  for(j=0; j+3<n; j+=4){
	if(first==1){
	  double *tmpB0 = &_b[(j+0)*size];
	  double *tmpB1 = &_b[(j+1)*size];
	  double *tmpB2 = &_b[(j+2)*size];
	  double *tmpB3 = &_b[(j+3)*size];
	  int y=0;
	  for(x=0;x<size;x+=1){
		tmpb[y++] = *tmpB0++;
		tmpb[y++] = *tmpB1++;
		tmpb[y++] = *tmpB2++;
		tmpb[y++] = *tmpB3++;
	  }
	}
	for(i=0; i+3<k; i+=4){
	  if(j==0){
		double *_cc;
		double *_tmpc = &tmpc[i*m];
		for(x=0;x<size;x++){
		  _cc = &_c[i+x*size];
		  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;
		}
	  }
	  // 4x4カーネルは同一
	  cpu_kernel_jik_avx_4x4(m,&_a[j*size+i],&tmpb[0],&tmpc[m*i],j,i);
	}
	if(i!=k){
	  for(; i<k; i++){
		cpu_kernel_jik_sse2_blk_inner_j4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	}
  }
  if(j!=n){
	for(; j<n; j++){
	  for(i=0; i+3<k; i+=4){
		cpu_kernel_jik_sse2_blk_inner_i4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	  if(i!=k){
		for(; i<k; i++){
		  cpu_kernel_jik_sse2_blk_inner_1(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
		}
	  }
	}
  }
}
// AVX、微調整版（v_a系の変更、性能はほぼ変わらない？）
void cpu_kernel_jik_avx2_blk(int size, int m, int n, int k, real *_a, real *_b, real *_c, int first);
#include "mm_base_cpu_avx.c"
void cpu_kernel_jik_avx2_4x4
(MYCONST int size, real * MYRESTRICT _a, MYCONST real * MYRESTRICT _b, MYCONST real * MYRESTRICT _c,
 MYCONST int j, MYCONST int i);
//void cpu_kernel_jik_avx2_4x4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_avx2(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size; j+=BLK_J){
	int i, n;
	n = min(size-j,BLK_J);
	for(i=0; i<size; i+=BLK_I){
	  int k;
	  k = min(size-i,BLK_I);
	  cpu_kernel_jik_avx2_blk(size, size,n,k,&_a[j*size+i],&_b[j*size],&_c[i],i==0);
	}
  }
}
void cpu_kernel_jik_avx2_blk(int size, int m, int n, int k, real *_a, real *_b, real *_c, int first)
{
  int i, j, x;
  double tmpc[m*k], tmpb[m*n];
  for(j=0; j+3<n; j+=4){
	  double *tmpB0 = &_b[(j+0)*size];
	  double *tmpB1 = &_b[(j+1)*size];
	  double *tmpB2 = &_b[(j+2)*size];
	  double *tmpB3 = &_b[(j+3)*size];
	  int y=0;
	  for(x=0;x<size;x+=1){
		tmpb[y++] = *tmpB0++;
		tmpb[y++] = *tmpB1++;
		tmpb[y++] = *tmpB2++;
		tmpb[y++] = *tmpB3++;
	  }
	for(i=0; i+3<k; i+=4){
	  if(j==0){
		double *_cc;
		double *_tmpc = &tmpc[i*m];
		for(x=0;x<size;x++){
		  _cc = &_c[i+x*size];
		  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;
		}
	  }
	  cpu_kernel_jik_avx2_4x4(m,&_a[j*size+i],&tmpb[0],&tmpc[m*i],j,i);
	}
	if(i!=k){
	  for(; i<k; i++){
		cpu_kernel_jik_sse2_blk_inner_j4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	}
  }
  if(j!=n){
	for(; j<n; j++){
	  for(i=0; i+3<k; i+=4){
		cpu_kernel_jik_sse2_blk_inner_i4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	  if(i!=k){
		for(; i<k; i++){
		  cpu_kernel_jik_sse2_blk_inner_1(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
		}
	  }
	}
  }
}
void cpu_kernel_jik_avx2_tmpb2_blk(int size, int m, int n, int k, real *_a, real *_b, real *_c, int first);
void cpu_kernel_jik_avx2_tmpb2(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size; j+=BLK_J){
	int i, n;
	n = min(size-j,BLK_J);
	for(i=0; i<size; i+=BLK_I){
	  int k;
	  k = min(size-i,BLK_I);
	  cpu_kernel_jik_avx2_tmpb2_blk(size, size,n,k,&_a[j*size+i],&_b[j*size],&_c[i],i==0);
	}
  }
}
void cpu_kernel_jik_avx2_tmpb2_blk(int size, int m, int n, int k, real *_a, real *_b, real *_c, int first)
{
  int i, j, x;
  double tmpc[m*k];
  static double tmpb[BSIZE]; // caution for parallel programs
  for(j=0; j+3<n; j+=4){
	if(first==1){
	  double *tmpB0 = &_b[(j+0)*size];
	  double *tmpB1 = &_b[(j+1)*size];
	  double *tmpB2 = &_b[(j+2)*size];
	  double *tmpB3 = &_b[(j+3)*size];
	  int y=0;
	  for(x=0;x<size;x+=1){
		tmpb[y++] = *tmpB0++;
		tmpb[y++] = *tmpB1++;
		tmpb[y++] = *tmpB2++;
		tmpb[y++] = *tmpB3++;
	  }
	}
	for(i=0; i+3<k; i+=4){
	  if(j==0){
		double *_cc;
		double *_tmpc = &tmpc[i*m];
		for(x=0;x<size;x++){
		  _cc = &_c[i+x*size];
		  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;
		}
	  }
	  cpu_kernel_jik_avx2_4x4(m,&_a[j*size+i],&tmpb[0],&tmpc[m*i],j,i);
	}
	if(i!=k){
	  for(; i<k; i++){
		cpu_kernel_jik_sse2_blk_inner_j4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	}
  }
  if(j!=n){
	for(; j<n; j++){
	  for(i=0; i+3<k; i+=4){
		cpu_kernel_jik_sse2_blk_inner_i4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	  if(i!=k){
		for(; i<k; i++){
		  cpu_kernel_jik_sse2_blk_inner_1(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
		}
	  }
	}
  }
}

// bのアドレス計算をインクリメントではなくした版
// やはり性能はほぼ同一のハズだが？
void cpu_kernel_jik_avx3_blk
(MYCONST int size, MYCONST int m, MYCONST int n, MYCONST int k,
 real * MYRESTRICT _a, MYCONST real * MYRESTRICT _b, MYCONST real * MYRESTRICT _c, MYCONST int first);
//void cpu_kernel_jik_avx3_blk(int size, int m, int n, int k, real *_a, real *_b, real *_c, int first);
#include "mm_base_cpu_avx.c"
void cpu_kernel_jik_avx3_4x4
(MYCONST int size, real * MYRESTRICT _a, MYCONST real * MYRESTRICT _b, MYCONST real * MYRESTRICT _c,
 MYCONST int j, MYCONST int i);
//void cpu_kernel_jik_avx3_4x4(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_avx3
(MYCONST int kernel, MYCONST int size, real * MYRESTRICT _a, MYCONST real * MYRESTRICT _b, MYCONST real * MYRESTRICT _c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size; j+=BLK_J){
	int i, n;
	n = min(size-j,BLK_J);
	for(i=0; i<size; i+=BLK_I){
	  int k;
	  k = min(size-i,BLK_I);
	  cpu_kernel_jik_avx3_blk(size, size,n,k,&_a[j*size+i],&_b[j*size],&_c[i],i==0);
	}
  }
}
void cpu_kernel_jik_avx3_blk
(MYCONST int size, MYCONST int m, MYCONST int n, MYCONST int k,
 real * MYRESTRICT _a, MYCONST real * MYRESTRICT _b, MYCONST real * MYRESTRICT _c, MYCONST int first)
//void cpu_kernel_jik_avx3_blk(int size, int m, int n, int k, real *_a, real *_b, real *_c, int first)
{
  int i, j, x;
  double tmpc[m*k], tmpb[m*n];
  for(j=0; j+3<n; j+=4){
    double *tmpB0 = &_b[(j+0)*size];
    double *tmpB1 = &_b[(j+1)*size];
    double *tmpB2 = &_b[(j+2)*size];
    double *tmpB3 = &_b[(j+3)*size];
	int y=0;
	for(x=0;x<size;x+=1){
	  tmpb[y++] = *tmpB0++;
	  tmpb[y++] = *tmpB1++;
	  tmpb[y++] = *tmpB2++;
	  tmpb[y++] = *tmpB3++;
	}
	for(i=0; i+3<k; i+=4){
	  if(j==0){
		double *_cc;
		double *_tmpc = &tmpc[i*m];
		for(x=0;x<size;x++){
		  _cc = &_c[i+x*size];
		  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;
		}
	  }
	  cpu_kernel_jik_avx3_4x4(m,&_a[j*size+i],&tmpb[0],&tmpc[m*i],j,i);
	}
	if(i!=k){
	  for(; i<k; i++){
		cpu_kernel_jik_sse2_blk_inner_j4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	}
  }
  if(j!=n){
	for(; j<n; j++){
	  for(i=0; i+3<k; i+=4){
		cpu_kernel_jik_sse2_blk_inner_i4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	  if(i!=k){
		for(; i<k; i++){
		  cpu_kernel_jik_sse2_blk_inner_1(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
		}
	  }
	}
  }
}
void cpu_kernel_jik_avx3_tmpb2_blk(int size, int m, int n, int k, real *_a, real *_b, real *_c, int first);
void cpu_kernel_jik_avx3_tmpb2(int kernel, int size, real *_a, real *_b, real *_c)
{
  int j;
#pragma omp parallel for
  for(j=0; j<size; j+=BLK_J){
	int i, n;
	n = min(size-j,BLK_J);
	for(i=0; i<size; i+=BLK_I){
	  int k;
	  k = min(size-i,BLK_I);
	  cpu_kernel_jik_avx3_tmpb2_blk(size, size,n,k,&_a[j*size+i],&_b[j*size],&_c[i],i==0);
	}
  }
}
void cpu_kernel_jik_avx3_tmpb2_blk(int size, int m, int n, int k, real *_a, real *_b, real *_c, int first)
{
  int i, j, x;
  double tmpc[m*k];
  static double tmpb[BSIZE]; // caution for parallel programs
  for(j=0; j+3<n; j+=4){
	if(first==1){
	  double *tmpB0 = &_b[(j+0)*size];
	  double *tmpB1 = &_b[(j+1)*size];
	  double *tmpB2 = &_b[(j+2)*size];
	  double *tmpB3 = &_b[(j+3)*size];
	  int y=0;
	  for(x=0;x<size;x+=1){
		tmpb[y++] = *tmpB0++;
		tmpb[y++] = *tmpB1++;
		tmpb[y++] = *tmpB2++;
		tmpb[y++] = *tmpB3++;
	  }
	}
	for(i=0; i+3<k; i+=4){
	  if(j==0){
		double *_cc;
		double *_tmpc = &tmpc[i*m];
		for(x=0;x<size;x++){
		  _cc = &_c[i+x*size];
		  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;
		}
	  }
	  cpu_kernel_jik_avx3_4x4(m,&_a[j*size+i],&tmpb[0],&tmpc[m*i],j,i);
	}
	if(i!=k){
	  for(; i<k; i++){
		cpu_kernel_jik_sse2_blk_inner_j4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	}
  }
  if(j!=n){
	for(; j<n; j++){
	  for(i=0; i+3<k; i+=4){
		cpu_kernel_jik_sse2_blk_inner_i4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	  if(i!=k){
		for(; i<k; i++){
		  cpu_kernel_jik_sse2_blk_inner_1(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
		}
	  }
	}
  }
}
// mic
void cpu_kernel_jik_avx3_mic_blk
(MYCONST int size, MYCONST int m, MYCONST int n, MYCONST int k,
 real * MYRESTRICT _a, MYCONST real * MYRESTRICT _b, MYCONST real * MYRESTRICT _c, MYCONST int first);
//void cpu_kernel_jik_avx3_mic_blk(int size, int m, int n, int k, real *_a, real *_b, real *_c, int first);
#include "mm_base_cpu_avx.c"
void cpu_kernel_jik_avx3_mic_8x8
(MYCONST int size, real * MYRESTRICT _a, MYCONST real * MYRESTRICT _b, MYCONST real * MYRESTRICT _c,
 MYCONST int j, MYCONST int i);
//void cpu_kernel_jik_avx3_mic_8x8(int size, real *_a, real *_b, real *_c, int j, int i);
void cpu_kernel_jik_avx3_mic
(MYCONST int kernel, MYCONST int size, real * MYRESTRICT _a, MYCONST real * MYRESTRICT _b, MYCONST real * MYRESTRICT _c)
{
  /*
  int j;
#pragma omp parallel for
  for(j=0; j<size; j+=BLK_J){
	int i, n;
	n = min(size-j,BLK_J);
	for(i=0; i<size; i+=BLK_I){
	  int k;
	  k = min(size-i,BLK_I);
	  cpu_kernel_jik_avx3_mic_blk(size, size,n,k,&_a[j*size+i],&_b[j*size],&_c[i],i==0);
	}
  }
  */
  int i, j, k, n;
#pragma omp parallel for private(i,k,n) collapse(2)
  for(j=0; j<size; j+=BLK_J){
	for(i=0; i<size; i+=BLK_I){
	  n = min(size-j,BLK_J);
	  k = min(size-i,BLK_I);
	  cpu_kernel_jik_avx3_mic_blk(size, size,n,k,&_a[j*size+i],&_b[j*size],&_c[i],i==0);
	}
  }
}
void cpu_kernel_jik_avx3_mic_blk
(MYCONST int size, MYCONST int m, MYCONST int n, MYCONST int k,
 real * MYRESTRICT _a, MYCONST real * MYRESTRICT _b, MYCONST real * MYRESTRICT _c, MYCONST int first)
{
  int i, j, x;
  double tmpc[m*k], tmpb[m*n];
  for(j=0; j+7<n; j+=8){
	double *tmpB0 = &_b[(j+0)*size];
	double *tmpB1 = &_b[(j+1)*size];
	double *tmpB2 = &_b[(j+2)*size];
	double *tmpB3 = &_b[(j+3)*size];
	double *tmpB4 = &_b[(j+4)*size];
	double *tmpB5 = &_b[(j+5)*size];
	double *tmpB6 = &_b[(j+6)*size];
	double *tmpB7 = &_b[(j+7)*size];
	int y=0;
	for(x=0;x<size;x+=1){
	  tmpb[y++] = *tmpB0++;	  tmpb[y++] = *tmpB1++;	  tmpb[y++] = *tmpB2++;	  tmpb[y++] = *tmpB3++;
	  tmpb[y++] = *tmpB4++;	  tmpb[y++] = *tmpB5++;	  tmpb[y++] = *tmpB6++;	  tmpb[y++] = *tmpB7++;
	}
	for(i=0; i+7<k; i+=8){
	  if(j==0){
		double *_cc;
		double *_tmpc = &tmpc[i*m];
		for(x=0;x<size;x++){
		  _cc = &_c[i+x*size];
		  *_tmpc ++ = *_cc++;	  *_tmpc ++ = *_cc++;	  *_tmpc ++ = *_cc++;	  *_tmpc ++ = *_cc++;
		  *_tmpc ++ = *_cc++;	  *_tmpc ++ = *_cc++;	  *_tmpc ++ = *_cc++;	  *_tmpc ++ = *_cc++;
		}
	  }
	  cpu_kernel_jik_avx3_mic_8x8(m,&_a[j*size+i],&tmpb[0],&tmpc[m*i],j,i);
	}
	/*
	if(i!=k){
	  for(; i<k; i++){
		cpu_kernel_jik_sse2_blk_inner_j4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	}
	*/
  }
  /*
  if(j!=n){
	for(; j<n; j++){
	  for(i=0; i+3<k; i+=4){
		cpu_kernel_jik_sse2_blk_inner_i4(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
	  }
	  if(i!=k){
		for(; i<k; i++){
		  cpu_kernel_jik_sse2_blk_inner_1(m,&_a[j*size+i],&_b[j*size],&_c[i],j,i);
		}
	  }
	}
  }
  */
}



void cpu_kernel(int kernel, int size, real *_a, real *_b, real *_c)
{
  switch(kernel){
  default: exit(-1); break;

  case 0:
	// ijkループ
	cpu_kernel_ijk(kernel, size, _a, _b, _c);
	break;
  case 1:
	// jikループ
	cpu_kernel_jik(kernel, size, _a, _b, _c);
	break;
  case 2:
	// jkiループ
	cpu_kernel_jki(kernel, size, _a, _b, _c);
	break;
  case 3:
	// 一時変数化
	cpu_kernel_jik_tmpsum(kernel, size, _a, _b, _c);
	break;
  case 4:
	// 最内ループのブロック化
	cpu_kernel_jik_blk(kernel, size, _a, _b, _c);
	break;
  case 5:
	// 最内ループのブロック化・別版
	cpu_kernel_jik_blk2(kernel, size, _a, _b, _c);
	break;
  case 6:
	// 中間ループもブロック化
	cpu_kernel_jik_blk3(kernel, size, _a, _b, _c);
	break;
  case 7:
	// 最外ループのアンローリング
	cpu_kernel_jik_unroll4(kernel, size, _a, _b, _c);
	break;
  case 8:
	// 最外ループのアンローリング＋一時変数（もちろん高速）
	cpu_kernel_jik_unroll4_tmp(kernel, size, _a, _b, _c);
	break;
  case 9:
	// 最外ループのアンローリング＋一時変数 another（もちろん低速）
	cpu_kernel_jik_unroll4_tmp2(kernel, size, _a, _b, _c);
	break;
  case 10:
	// jkiループ＋最外ループアンローリング
	cpu_kernel_jki_unroll4(kernel, size, _a, _b, _c);
	break;
  case 11:
	// jkiループ＋最外ループアンローリング＋一時変数＋共通項まとめ
	cpu_kernel_jki_unroll4_tmp(kernel, size, _a, _b, _c);
	break;
  case 12:
	// ijkループ＋最外ループアンローリング
	cpu_kernel_ijk_unroll4(kernel, size, _a, _b, _c);
	break;
  case 13:
	// ijkループ＋最外ループアンローリング＋一時変数
	cpu_kernel_ijk_unroll4_tmp(kernel, size, _a, _b, _c);
	break;
  case 14:
	// ijkループ＋最外ループアンローリング＋一時変数＋共通項まとめ
	cpu_kernel_ijk_unroll4_tmp2(kernel, size, _a, _b, _c);
	break;
  case 15:
	// jikループ＋最外ループアンローリング＋一時変数＋共通項まとめ＋ポインタ化
	cpu_kernel_jik_unroll4_tmp2_ptr(kernel, size, _a, _b, _c);
	break;
  case 16:
	// register変数化、あまり意味はないと思うのだが、サイズ次第で優位な差が？
	cpu_kernel_jik_unroll4_tmp2_ptr_reg(kernel, size, _a, _b, _c);
	break;
  case 17:
	// jikループ＋最外ループアンローリング＋一時変数＋共通項まとめ
	// ＋ポインタ化＋register＋最内ループアンローリング
	cpu_kernel_jik_unroll4_tmp2_ptr_reg_unroll4(kernel, size, _a, _b, _c);
	break;
  case 18:
	// jikループ＋最外ループアンローリング＋一時変数＋共通項まとめ
	// ＋ポインタ化＋register＋最内ループアンローリング＋インクリメント潰し
	cpu_kernel_jik_unroll4_tmp2_ptr_reg_unroll4_noincl(kernel, size, _a, _b, _c);
	break;
  case 19:
	// jikループ＋最外ループアンローリング＋一時変数＋共通項まとめ
	// ＋ポインタ化＋register＋最内ループアンローリング＋インクリメント潰し＋複製
	cpu_kernel_jik_unroll4_tmp2_ptr_reg_unroll4_noincl_clone(kernel, size, _a, _b, _c);
	break;

	// 最内ループの関数化

  case 20:
	// jik、最内ループ関数化、外部4x4ブロック化
	cpu_kernel_jik_func_out16(kernel, size, _a, _b, _c);
	break;
  case 21:
	// jik、最内ループ関数化＋アンローリング、外部4X4ブロック化
	cpu_kernel_jik_func_out16_unroll4(kernel, size, _a, _b, _c);
	break;
  case 22:
	// jik、最内ループ関数化、外側4x4ブロック化、外部制御
	cpu_kernel_jik_func_out16_outcntl(kernel, size, _a, _b, _c);
	break;
  case 23:
	// jik、最内ループ関数化、外側4x4ブロック化、外部制御べた書き
	cpu_kernel_jik_func_out16_flat16(kernel, size, _a, _b, _c);
	break;
  case 24:
	// jik、最内ループ関数化、外側4x4ブロック化、外部制御べた書き、関数内アンローリング
	cpu_kernel_jik_func_out16_flat16_unroll4(kernel, size, _a, _b, _c);
	break;
  case 25:
	// jik、最内ループ関数化、外部4x4ブロック化、計算順序微調整版
	cpu_kernel_jik_func_out16_mod(kernel, size, _a, _b, _c);
	break;
  case 26:
	// jik、最内ループ関数化、外部4x4ブロック化、計算順序微調整版2
	cpu_kernel_jik_func_out16_mod2(kernel, size, _a, _b, _c);
	break;

	// SSE命令
  case 27:
	// ベクトル命令版
	// 事実上cpu_kernel_jik_func_out16_modのSSE版
	cpu_kernel_jik_sse(kernel, size, _a, _b, _c);
	break;
  case 28:
	// 外側でオフセット計算版
	cpu_kernel_jik_sse2(kernel, size, _a, _b, _c);
	break;
  case 29:
	// ブロック版
	cpu_kernel_jik_sse2_blk(kernel, size, _a, _b, _c);
	break;
  case 30:
	// ブロック版＋配列Cの不連続アクセスを一時変数で解消
	// 4x4以外はとりあえず放置（解消無し）
	cpu_kernel_jik_sse2_blk_tmpc(kernel, size, _a, _b, _c);
	break;
  case 31:
	// ブロック版＋配列Cの不連続アクセスを一時変数で解消、格納処理をまとめる
	// 4x4以外はとりあえず放置（解消無し）
	cpu_kernel_jik_sse2_blk_tmpc2(kernel, size, _a, _b, _c);
	break;
  case 32:
	// ブロック版＋配列Cの不連続アクセスを一時変数で解消、格納処理をまとめる
	// Bも一時変数に格納
	// 4x4以外はとりあえず放置（解消無し）
	cpu_kernel_jik_sse2_blk_tmpc2_tmpb(kernel, size, _a, _b, _c);
	break;
  case 33:
	// ブロック版＋配列Cの不連続アクセスを一時変数で解消、格納処理をまとめる
	// Bも一時変数に格納、格納を最初だけに制限（並列実行時に注意）
	// 4x4以外はとりあえず放置（解消無し）
	cpu_kernel_jik_sse2_blk_tmpc2_tmpb2(kernel, size, _a, _b, _c);
	break;

  case 40:
	// AVX
	cpu_kernel_jik_avx(kernel, size, _a, _b, _c);
	break;
  case 41:
	// AVX
	// OpenMP非対応
	cpu_kernel_jik_avx_tmpb2(kernel, size, _a, _b, _c);
	break;
  case 42:
	// AVX、微調整版（v_a系の変更、性能はほぼ変わらない？）
	cpu_kernel_jik_avx2(kernel, size, _a, _b, _c);
	break;
  case 43:
	// AVX、微調整版（v_a系の変更、性能はほぼ変わらない？）
	// OpenMP非対応
	cpu_kernel_jik_avx2_tmpb2(kernel, size, _a, _b, _c);
	break;
  case 44:
	// bのアドレス計算をインクリメントではなくした版
	// やはり性能はほぼ同一のハズだが？
	cpu_kernel_jik_avx3(kernel, size, _a, _b, _c);
	break;
  case 45:
	// bのアドレス計算をインクリメントではなくした版
	// やはり性能はほぼ同一のハズだが？
	// OpenMP非対応
	cpu_kernel_jik_avx3_tmpb2(kernel, size, _a, _b, _c);
	break;

  case 46:
    // AVX3 for mic
    // とりあえず端数処理無し
    cpu_kernel_jik_avx3_mic(kernel, size, _a, _b, _c);
    break;

#ifndef _NO_MKL
  case 50:
    // MKL
    cblas_dgemm
      (
       CblasRowMajor, CblasNoTrans, CblasNoTrans,
       size, size, size, 1.0,
       _b, size,
       _c, size, 1.0, _a, size
       );
    break;

  case 51:
	{
	  double one = 1.0;
    dgemm_
      (
	   "N", "N",
       &size, &size, &size, &one,
       _b, &size,
       _c, &size, &one, _a, &size
       );
	}
    break;
#endif
  }
}

// ******** ******** ******** ********
// main
// ******** ******** ******** ********
int main(int argc, char** argv)
{
  int i;
  printf("command:");for(i=0;i<argc;i++)printf(" %s",argv[i]);printf("\n");
  if(checkArgs(argc,argv))return -1;

  MyPrintf("size %d\n", SIZE);
  srand(RANDSEED);

  //MyPrintf("initialize...");
  int t,g;
#ifdef _USE_AVX3
  //g_A = (double*)mmap(NULL, sizeof(double)*(SIZE*SIZE)+4096, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_SHARED|MAP_POPULATE, -1, 0);
  //g_B = (double*)mmap(NULL, sizeof(double)*(SIZE*SIZE)+4096, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_SHARED|MAP_POPULATE, -1, 0);
  //g_C = (double*)mmap(NULL, sizeof(double)*(SIZE*SIZE)+4096, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_SHARED|MAP_POPULATE, -1, 0);
  g_A = (double*)_malloc_2M(sizeof(double)*SIZE*SIZE);
  g_B = (double*)_malloc_2M(sizeof(double)*SIZE*SIZE);
  g_C = (double*)_malloc_2M(sizeof(double)*SIZE*SIZE);
#else
  g_A = (real*)malloc(sizeof(real)*SIZE*SIZE);
  g_B = (real*)malloc(sizeof(real)*SIZE*SIZE);
  g_C = (real*)malloc(sizeof(real)*SIZE*SIZE);
#ifndef _NO_MKL
#ifdef _NO_MIC_MKL
  mkl_mic_disable();
#endif
#endif
#endif
  //MyPrintf("A %x\nB %x\nC %x\n", g_A, g_B, g_C);

  for(g=0; g<SIZE; g++){
	for(t=0; t<SIZE; t++){
	  g_A[g*SIZE+t] = frand();
	  g_B[g*SIZE+t] = frand();
	  g_C[g*SIZE+t] = frand();
	}
  }
  //MyPrintf("done\n");

  struct timeval tBegin, tEnd;
  struct timezone tz;
  double *dSecs;
  dSecs = (double*)malloc(sizeof(double)*LOOPS);
  double dBegin, dEnd;
  cpu_kernel(KERNEL, SIZE, g_A, g_B, g_C);
  for(i=0;i<LOOPS;i++){
	gettimeofday(&tBegin, &tz);
	cpu_kernel(KERNEL, SIZE, g_A, g_B, g_C);
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
  MyPrintf("performance: kernel %d, size %d^2, total %f sec, average %f sec, min %f sec, max %f sec, %f Mflops",
		 KERNEL, SIZE, dSecSum, dSecSum/(double)LOOPS, dSecMin, dSecMax,
		 (double)(LOOPS*2.0*(double)SIZE*(double)SIZE*(double)SIZE)/dSecSum/1000.0/1000.0);

  {
	MyPrintf(" sum");
	double tmp=0.0;
#pragma omp parallel for reduction(+:tmp)
	for(i=0; i<10; i++){
	  tmp += g_A[i];
	}
	MyPrintf(" %f",tmp);
  }
  MyPrintf("\n");
  {
	MyPrintf("random sampling:");
	for(i=0; i<10; i++){
	  int n = rand()%(SIZE*SIZE);
	  MyPrintf(" %.2f", g_A[n]);
	}
	MyPrintf("\n");
  }
  /*
  {
	int i, j;
	FILE *F;
	char filename[16];
	snprintf(filename, 16, "A_base_k%d.txt", KERNEL);
	F = fopen(filename, "w");
	for(i=0; i<SIZE; i++){
	  for(j=0; j<SIZE; j++){
		fprintf(F, " %.2f", g_A[i*SIZE+j]);
	  }
	  fprintf(F, "\n");
	}
	fclose(F);
  }
  */
#ifdef _USE_AVX3
  munmap(g_A, (sizeof(double)*(SIZE*SIZE)+4096));
  munmap(g_B, (sizeof(double)*(SIZE*SIZE)+4096));
  munmap(g_C, (sizeof(double)*(SIZE*SIZE)+4096));
#else
  free(g_A);
  free(g_B);
  free(g_C);
#endif
  if(fout!=NULL)fclose(fout);
  return 0;
}
