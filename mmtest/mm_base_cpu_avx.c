#ifndef _AVX_C
#define _AVX_C
void cpu_kernel_jik_avx_4x4
(MYCONST int size, real * MYRESTRICT _a, MYCONST real * MYRESTRICT _b, MYCONST real * MYRESTRICT _c,
 MYCONST int j, MYCONST int i)
{
#ifdef _USE_AVX
  int k;
  v4df_t v_a0, v_a1, v_a2, v_a3;
  v4df_t v_b0, v_b1, v_b2, v_b3;
  v4df_t v_c;
  v_a0.v = _mm256_setzero_pd();
  v_a1.v = _mm256_setzero_pd();
  v_a2.v = _mm256_setzero_pd();
  v_a3.v = _mm256_setzero_pd();
  for(k=0; k<size; k++){
	v_c.v = _mm256_load_pd((double*)_c); _c+=4;
	v_b0.v = _mm256_broadcast_sd((double*)_b++);
	v_b1.v = _mm256_broadcast_sd((double*)_b++);
	v_b2.v = _mm256_broadcast_sd((double*)_b++);
	v_b3.v = _mm256_broadcast_sd((double*)_b++);
	v_a0.v = _mm256_add_pd(v_a0.v, _mm256_mul_pd(v_b0.v, v_c.v));
	v_a1.v = _mm256_add_pd(v_a1.v, _mm256_mul_pd(v_b1.v, v_c.v));
	v_a2.v = _mm256_add_pd(v_a2.v, _mm256_mul_pd(v_b2.v, v_c.v));
	v_a3.v = _mm256_add_pd(v_a3.v, _mm256_mul_pd(v_b3.v, v_c.v));
  }
  _a[0*size] += v_a0.d[0];
  _a[1*size] += v_a1.d[0];
  _a[2*size] += v_a2.d[0];
  _a[3*size] += v_a3.d[0];
  _a[0*size+1] += v_a0.d[1];
  _a[1*size+1] += v_a1.d[1];
  _a[2*size+1] += v_a2.d[1];
  _a[3*size+1] += v_a3.d[1];
  _a[0*size+2] += v_a0.d[2];
  _a[1*size+2] += v_a1.d[2];
  _a[2*size+2] += v_a2.d[2];
  _a[3*size+2] += v_a3.d[2];
  _a[0*size+3] += v_a0.d[3];
  _a[1*size+3] += v_a1.d[3];
  _a[2*size+3] += v_a2.d[3];
  _a[3*size+3] += v_a3.d[3];
#endif
}


void cpu_kernel_jik_avx2_4x4
(MYCONST int size, real * MYRESTRICT _a, MYCONST real * MYRESTRICT _b, MYCONST real * MYRESTRICT _c,
 MYCONST int j, MYCONST int i)
{
#ifdef _USE_AVX
  int k;
  v4df_t v_a0, v_a1, v_a2, v_a3;
  v4df_t v_b0, v_b1, v_b2, v_b3;
  v4df_t v_c;
  v_a0.v = _mm256_load_pd((double*)_a);
  v_a1.v = _mm256_load_pd((double*)_a+4); // are?
  v_a2.v = _mm256_load_pd((double*)_a+8);
  v_a3.v = _mm256_load_pd((double*)_a+12);
  for(k=0; k<size; k++){
	v_c.v = _mm256_load_pd((double*)_c); _c+=4;
	v_b0.v = _mm256_broadcast_sd((double*)_b++);
	v_b1.v = _mm256_broadcast_sd((double*)_b++);
	v_b2.v = _mm256_broadcast_sd((double*)_b++);
	v_b3.v = _mm256_broadcast_sd((double*)_b++);
	v_a0.v = _mm256_add_pd(v_a0.v, _mm256_mul_pd(v_b0.v, v_c.v));
	v_a1.v = _mm256_add_pd(v_a1.v, _mm256_mul_pd(v_b1.v, v_c.v));
	v_a2.v = _mm256_add_pd(v_a2.v, _mm256_mul_pd(v_b2.v, v_c.v));
	v_a3.v = _mm256_add_pd(v_a3.v, _mm256_mul_pd(v_b3.v, v_c.v));
  }
  _mm256_store_pd(&_a[0*size], v_a0.v);
  _mm256_store_pd(&_a[1*size], v_a1.v);
  _mm256_store_pd(&_a[2*size], v_a2.v);
  _mm256_store_pd(&_a[3*size], v_a3.v);
#endif
}


void cpu_kernel_jik_avx3_4x4
(MYCONST int size, real * MYRESTRICT _a, MYCONST real * MYRESTRICT _b, MYCONST real * MYRESTRICT _c,
 MYCONST int j, MYCONST int i)
{
#ifdef _USE_AVX
  int k;
  v4df_t v_a0, v_a1, v_a2, v_a3;
  v4df_t v_b0, v_b1, v_b2, v_b3;
  v4df_t v_c;
  v_a0.v = _mm256_load_pd((double*)_a);
  v_a1.v = _mm256_load_pd((double*)_a+4);
  v_a2.v = _mm256_load_pd((double*)_a+8);
  v_a3.v = _mm256_load_pd((double*)_a+12);
  for(k=0; k<size; k++){
	v_c.v = _mm256_load_pd((double*)_c); _c+=4;
	v_b0.v = _mm256_broadcast_sd((double*)_b);
	v_b1.v = _mm256_broadcast_sd((double*)_b+1);
	v_b2.v = _mm256_broadcast_sd((double*)_b+2);
	v_b3.v = _mm256_broadcast_sd((double*)_b+3); _b+=4;
	v_a0.v = _mm256_add_pd(v_a0.v, _mm256_mul_pd(v_b0.v, v_c.v));
	v_a1.v = _mm256_add_pd(v_a1.v, _mm256_mul_pd(v_b1.v, v_c.v));
	v_a2.v = _mm256_add_pd(v_a2.v, _mm256_mul_pd(v_b2.v, v_c.v));
	v_a3.v = _mm256_add_pd(v_a3.v, _mm256_mul_pd(v_b3.v, v_c.v));
  }
  _mm256_store_pd(&_a[0*size], v_a0.v);
  _mm256_store_pd(&_a[1*size], v_a1.v);
  _mm256_store_pd(&_a[2*size], v_a2.v);
  _mm256_store_pd(&_a[3*size], v_a3.v);
#endif
}

#if 0
// test
void cpu_kernel_jik_avx3_mic_8x8(int size, real *_a, real *_b, real *_c, int j, int i)
{
#ifdef _USE_AVX3
  int k;
  v8df_t v_a0, v_a1, v_a2, v_a3, v_a4, v_a5, v_a6, v_a7;
  v8df_t v_b0, v_b1, v_b2, v_b3, v_b4, v_b5, v_b6, v_b7;
  v8df_t v_c;
  v_a0.v = _mm512_setzero_pd();
  v_a1.v = _mm512_setzero_pd();
  v_a2.v = _mm512_setzero_pd();
  v_a3.v = _mm512_setzero_pd();
  v_a4.v = _mm512_setzero_pd();
  v_a5.v = _mm512_setzero_pd();
  v_a6.v = _mm512_setzero_pd();
  v_a7.v = _mm512_setzero_pd();
  /*
  for(k=0; k<size; k++){
    v_c.v  = _mm512_load_pd((double*)_c); _c+=8;
    v_b0.v = _mm512_extload_pd((double*)_b+0, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
    v_b1.v = _mm512_extload_pd((double*)_b+1, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
    v_b2.v = _mm512_extload_pd((double*)_b+2, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
    v_b3.v = _mm512_extload_pd((double*)_b+3, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
    v_b4.v = _mm512_extload_pd((double*)_b+4, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
    v_b5.v = _mm512_extload_pd((double*)_b+5, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
    v_b6.v = _mm512_extload_pd((double*)_b+6, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
    v_b7.v = _mm512_extload_pd((double*)_b+7, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
    _b+=8;
    v_a0.v = _mm512_add_pd(v_a0.v, _mm512_mul_pd(v_b0.v, v_c.v));
    v_a1.v = _mm512_add_pd(v_a1.v, _mm512_mul_pd(v_b1.v, v_c.v));
    v_a2.v = _mm512_add_pd(v_a2.v, _mm512_mul_pd(v_b2.v, v_c.v));
    v_a3.v = _mm512_add_pd(v_a3.v, _mm512_mul_pd(v_b3.v, v_c.v));
    v_a4.v = _mm512_add_pd(v_a4.v, _mm512_mul_pd(v_b4.v, v_c.v));
    v_a5.v = _mm512_add_pd(v_a5.v, _mm512_mul_pd(v_b5.v, v_c.v));
    v_a6.v = _mm512_add_pd(v_a6.v, _mm512_mul_pd(v_b6.v, v_c.v));
    v_a7.v = _mm512_add_pd(v_a7.v, _mm512_mul_pd(v_b7.v, v_c.v));
  }
  */
  _a[0*size] += v_a0.d[0];
  _a[1*size] += v_a1.d[0];
  _a[2*size] += v_a2.d[0];
  _a[3*size] += v_a3.d[0];
  _a[4*size] += v_a4.d[0];
  _a[5*size] += v_a5.d[0];
  _a[6*size] += v_a6.d[0];
  _a[7*size] += v_a7.d[0];
  _a[0*size+1] += v_a0.d[1];
  _a[1*size+1] += v_a1.d[1];
  _a[2*size+1] += v_a2.d[1];
  _a[3*size+1] += v_a3.d[1];
  _a[4*size+1] += v_a4.d[1];
  _a[5*size+1] += v_a5.d[1];
  _a[6*size+1] += v_a6.d[1];
  _a[7*size+1] += v_a7.d[1];
  _a[0*size+2] += v_a0.d[2];
  _a[1*size+2] += v_a1.d[2];
  _a[2*size+2] += v_a2.d[2];
  _a[3*size+2] += v_a3.d[2];
  _a[4*size+2] += v_a4.d[2];
  _a[5*size+2] += v_a5.d[2];
  _a[6*size+2] += v_a6.d[2];
  _a[7*size+2] += v_a7.d[2];
  _a[0*size+3] += v_a0.d[3];
  _a[1*size+3] += v_a1.d[3];
  _a[2*size+3] += v_a2.d[3];
  _a[3*size+3] += v_a3.d[3];
  _a[4*size+3] += v_a4.d[3];
  _a[5*size+3] += v_a5.d[3];
  _a[6*size+3] += v_a6.d[3];
  _a[7*size+3] += v_a7.d[3];
  _a[0*size+4] += v_a0.d[4];
  _a[1*size+4] += v_a1.d[4];
  _a[2*size+4] += v_a2.d[4];
  _a[3*size+4] += v_a3.d[4];
  _a[4*size+4] += v_a4.d[4];
  _a[5*size+4] += v_a5.d[4];
  _a[6*size+4] += v_a6.d[4];
  _a[7*size+4] += v_a7.d[4];
  _a[0*size+5] += v_a0.d[5];
  _a[1*size+5] += v_a1.d[5];
  _a[2*size+5] += v_a2.d[5];
  _a[3*size+5] += v_a3.d[5];
  _a[4*size+5] += v_a4.d[5];
  _a[5*size+5] += v_a5.d[5];
  _a[6*size+5] += v_a6.d[5];
  _a[7*size+5] += v_a7.d[5];
  _a[0*size+6] += v_a0.d[6];
  _a[1*size+6] += v_a1.d[6];
  _a[2*size+6] += v_a2.d[6];
  _a[3*size+6] += v_a3.d[6];
  _a[4*size+6] += v_a4.d[6];
  _a[5*size+6] += v_a5.d[6];
  _a[6*size+6] += v_a6.d[6];
  _a[7*size+6] += v_a7.d[6];
  _a[0*size+7] += v_a0.d[7];
  _a[1*size+7] += v_a1.d[7];
  _a[2*size+7] += v_a2.d[7];
  _a[3*size+7] += v_a3.d[7];
  _a[4*size+7] += v_a4.d[7];
  _a[5*size+7] += v_a5.d[7];
  _a[6*size+7] += v_a6.d[7];
  _a[7*size+7] += v_a7.d[7];
#endif
}
#else
void cpu_kernel_jik_avx3_mic_8x8
(MYCONST int size, real * MYRESTRICT _a, MYCONST real * MYRESTRICT _b, MYCONST real * MYRESTRICT _c,
 MYCONST int j, MYCONST int i)
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
#endif
#endif
