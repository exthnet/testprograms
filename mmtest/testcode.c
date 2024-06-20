#define _malloc_2M(X) mmap(NULL, (X+4096), PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_SHARED|MAP_HUGETLB|MAP_POPULATE, -1, 0)
#define BLK_J 32
#define BLK_I 32
#include <sys/mman.h>
#include <immintrin.h> // AVX
typedef union
{
  __m512d v;
  double d[8];
}v8df_t;

void cpu_kernel_jik_avx3_mic(int kernel, int size, real *_a, real *_b, real *_c)
{
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
void cpu_kernel_jik_avx3_mic_blk(int size, int m, int n, int k, real *_a, real *_b, real *_c, int first)
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
      tmpb[y++] = *tmpB0++;   tmpb[y++] = *tmpB1++;   tmpb[y++] = *tmpB2++;   tmpb[y++] = *tmpB3++;
      tmpb[y++] = *tmpB4++;   tmpb[y++] = *tmpB5++;   tmpb[y++] = *tmpB6++;   tmpb[y++] = *tmpB7++;
    }
    for(i=0; i+7<k; i+=8){
      if(j==0){
	double *_cc;
	double *_tmpc = &tmpc[i*m];
	for(x=0;x<size;x++){
	  _cc = &_c[i+x*size];
	  *_tmpc ++ = *_cc++;     *_tmpc ++ = *_cc++;     *_tmpc ++ = *_cc++;     *_tmpc ++ = *_cc++;
	  *_tmpc ++ = *_cc++;     *_tmpc ++ = *_cc++;     *_tmpc ++ = *_cc++;     *_tmpc ++ = *_cc++;
	}
      }
      cpu_kernel_jik_avx3_mic_8x8(m,&_a[j*size+i],&tmpb[0],&tmpc[m*i],j,i);
    }
  }
}

void cpu_kernel_jik_avx3_mic_8x8(int size, real *_a, real *_b, real *_c, int j, int i)
{
#ifdef _USE_AVX3
  int k;
  v8df_t v_a0, v_a1, v_a2, v_a3, v_a4, v_a5, v_a6, v_a7;
  v8df_t v_b0, v_b1, v_b2, v_b3, v_b4, v_b5, v_b6, v_b7;
  v8df_t v_c;
  v_a0.v = _mm512_extload_pd((double*)_a, _MM_UPCONV_PD_NONE, _MM_BROADCAST64_NONE, _MM_HINT_NTA);
  v_a1.v = _mm512_extload_pd((double*)&_a[1*size], _MM_UPCONV_PD_NONE, _MM_BROADCAST64_NONE, _MM_HINT_NTA);
  v_a2.v = _mm512_extload_pd((double*)&_a[2*size], _MM_UPCONV_PD_NONE, _MM_BROADCAST64_NONE, _MM_HINT_NTA);
  v_a3.v = _mm512_extload_pd((double*)&_a[3*size], _MM_UPCONV_PD_NONE, _MM_BROADCAST64_NONE, _MM_HINT_NTA);
  v_a4.v = _mm512_extload_pd((double*)&_a[4*size], _MM_UPCONV_PD_NONE, _MM_BROADCAST64_NONE, _MM_HINT_NTA);
  v_a5.v = _mm512_extload_pd((double*)&_a[5*size], _MM_UPCONV_PD_NONE, _MM_BROADCAST64_NONE, _MM_HINT_NTA);
  v_a6.v = _mm512_extload_pd((double*)&_a[6*size], _MM_UPCONV_PD_NONE, _MM_BROADCAST64_NONE, _MM_HINT_NTA);
  v_a7.v = _mm512_extload_pd((double*)&_a[7*size], _MM_UPCONV_PD_NONE, _MM_BROADCAST64_NONE, _MM_HINT_NTA);
  for(k=0; k<size; k++){
    _mm_prefetch((char*)(_c+8*2), _MM_HINT_T0);
    _mm_prefetch((char*)(_c+8*8), _MM_HINT_T2);
    v_c.v  = _mm512_load_pd((double*)_c); _c+=8;
    _mm_prefetch((char*)(_b+8*2), _MM_HINT_T0);
    _mm_prefetch((char*)(_b+8*8), _MM_HINT_T2);
    v_b0.v = _mm512_extload_pd((double*)_b+0, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
    v_b1.v = _mm512_extload_pd((double*)_b+1, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
    v_b2.v = _mm512_extload_pd((double*)_b+2, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
    v_b3.v = _mm512_extload_pd((double*)_b+3, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
    v_b4.v = _mm512_extload_pd((double*)_b+4, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
    v_b5.v = _mm512_extload_pd((double*)_b+5, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
    v_b6.v = _mm512_extload_pd((double*)_b+6, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
    v_b7.v = _mm512_extload_pd((double*)_b+7, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
    _b+=8;
    _mm_prefetch((char*)(_a+8*2), _MM_HINT_T0);
    _mm_prefetch((char*)(_a+8*8), _MM_HINT_T2);
    /*
    v_a0.v = _mm512_add_pd(v_a0.v, _mm512_mul_pd(v_b0.v, v_c.v));
    v_a1.v = _mm512_add_pd(v_a1.v, _mm512_mul_pd(v_b1.v, v_c.v));
    v_a2.v = _mm512_add_pd(v_a2.v, _mm512_mul_pd(v_b2.v, v_c.v));
    v_a3.v = _mm512_add_pd(v_a3.v, _mm512_mul_pd(v_b3.v, v_c.v));
    v_a4.v = _mm512_add_pd(v_a4.v, _mm512_mul_pd(v_b4.v, v_c.v));
    v_a5.v = _mm512_add_pd(v_a5.v, _mm512_mul_pd(v_b5.v, v_c.v));
    v_a6.v = _mm512_add_pd(v_a6.v, _mm512_mul_pd(v_b6.v, v_c.v));
    v_a7.v = _mm512_add_pd(v_a7.v, _mm512_mul_pd(v_b7.v, v_c.v));
    */
    v_a0.v = _mm512_fmadd_pd(v_b0.v, v_c.v, v_a0.v);
    v_a1.v = _mm512_fmadd_pd(v_b1.v, v_c.v, v_a1.v);
    v_a2.v = _mm512_fmadd_pd(v_b2.v, v_c.v, v_a2.v);
    v_a3.v = _mm512_fmadd_pd(v_b3.v, v_c.v, v_a3.v);
    v_a4.v = _mm512_fmadd_pd(v_b4.v, v_c.v, v_a4.v);
    v_a5.v = _mm512_fmadd_pd(v_b5.v, v_c.v, v_a5.v);
    v_a6.v = _mm512_fmadd_pd(v_b6.v, v_c.v, v_a6.v);
    v_a7.v = _mm512_fmadd_pd(v_b7.v, v_c.v, v_a7.v);
  }
  _mm512_store_pd(&_a[0*size], v_a0.v);
  _mm512_store_pd(&_a[1*size], v_a1.v);
  _mm512_store_pd(&_a[2*size], v_a2.v);
  _mm512_store_pd(&_a[3*size], v_a3.v);
  _mm512_store_pd(&_a[4*size], v_a4.v);
  _mm512_store_pd(&_a[5*size], v_a5.v);
  _mm512_store_pd(&_a[6*size], v_a6.v);
  _mm512_store_pd(&_a[7*size], v_a7.v);
#endif
}
