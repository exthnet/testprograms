void acckernel(int N, float *C, float *A, float *B)
{
#pragma acc kernels deviceptr(A,B,C)
#pragma acc loop independent
  for(int i=0; i<N; i++){
	C[i] += A[i] * B[i];
  }
}
