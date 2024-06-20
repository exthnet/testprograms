void cpu_kernel_jik_sse2_blk_tmpc_inner_4x4(int size, real *_a, real *_b, real *_c, int j, int i)
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
	v_c0a0b.v = _mm_load_pd((double*)&_c[k*4]);
	v_c0c0d.v = _mm_load_pd((double*)&_c[k*4+2]);
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
