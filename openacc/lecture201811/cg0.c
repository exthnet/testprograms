#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Congugate Gradient Method: solve Ax=b

double frand(double d)
{
  return (double)(rand()%100+1.0)/100.0 * d;
}

int main(int argc, char **argv)
{
  int i, j;
  int N = 10;
  int iter, maxiter=100;
  int randseed=0;
  double alpha, beta;
  double *r, *p, *Ap, *Ax, rr, rr0, rp, norm, eps=1.0e-10;
  double t1, t2;
  // testdata
#if 0
  double A[100] = {
	5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	2.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 2.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 2.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 2.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 2.0, 5.0, 2.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 5.0, 2.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 5.0, 2.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 5.0, 2.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 5.0};
  double b[10] = {3.0, 1.0, 4.0, 0.0, 5.0, -1.0, 6.0, -2.0, 7.0, -15.0};
  double x[10] = {1.0, 1.0, 1.0, 1.0, 1.0,  1.0, 1.0,  1.0, 1.0,   1.0};
  double xx[10] = {1.0, 1.0, 1.0, 1.0, 1.0,  1.0, 1.0,  1.0, 1.0,   1.0};
#else
  if(argc>1){N=atoi(argv[1]); printf("N=%d\n",N);}
  if(argc>2)randseed=atoi(argv[1]);

  double *A, *b, *x;
  double *xx;
  srand(randseed);
  A = (double*)malloc(sizeof(double)*N*N);
  b = (double*)malloc(sizeof(double)*N);
  x = (double*)malloc(sizeof(double)*N);
  xx = (double*)malloc(sizeof(double)*N);
#if 0
  for(i=0;i<N;i++){
	for(j=i;j<N;j++){
	  double d = frand((double)N);
	  A[i*N+j]=d;
	  A[j*N+i]=d;
	}
  }
#else
  for(i=0;i<N;i++){
	for(j=0;j<N;j++){
	  if(i==j)A[i*N+j] = (double)(N-1);
	  else A[i*N+j] = -1.0;
	}
  }
#endif
  for(i=0;i<N;i++)x[i]=frand((double)N);
  for(i=0;i<N;i++){
	b[i] = 0.0;
	for(j=0;j<N;j++){
	  b[i]+=A[i*N+j]*x[j];
	}
  }
  for(i=0;i<N;i++)xx[i]=x[i];
  for(i=0;i<N;i++)x[i]=0.0;
#endif
  r = (double*)malloc(sizeof(double)*N);
  p = (double*)malloc(sizeof(double)*N);
  Ap = (double*)malloc(sizeof(double)*N);
  Ax = (double*)malloc(sizeof(double)*N);

  // check data
#if 1
  printf("A:\n");
  for(i=0;i<N;i++){
	for(j=0;j<N;j++){
	  printf(" %f", A[i*N+j]);
	}
	printf("\n");
  }
  printf("b:\n");
  for(i=0;i<N;i++){
	printf(" %f\n", b[i]);
  }
#endif

  // r_0 := b - A * x_0
  for(i=0;i<N;i++){
	Ax[i] = 0.0;
	for(j=0;j<N;j++){
	  Ax[i] += A[i*N+j]*x[j];
	}
  }
  for(i=0;i<N;i++){
	r[i]=b[i]-Ax[i];
  }

  // p_0 = r_0
  for(i=0;i<N;i++){
	p[i]=r[i];
  }

  // rr = rt_k * r_k
  for(i=0;i<N;i++){
	rr=r[i]*r[i];
  }

  t1 = omp_get_wtime();
  // for i = 0, 1, 2, ..., until convergence do
  for(iter=1; iter<=maxiter; iter++){
	printf("iter %d ", iter);
	// a_k = (rt_k * p_k) / (pt_k * A * p_k)
	for(i=0;i<N;i++){
	  Ap[i] = 0.0;
	  for(j=0;j<N;j++){
		Ap[i] += A[i*N+j]*p[j];
	  }
	}
	rp = 0.0;
	for(i=0;i<N;i++){
	  rp+=r[i]*p[i];
	}
	alpha = 0.0;
	for(i=0;i<N;i++){
	  alpha+=p[i]*Ap[i];
	}
	alpha = rp / alpha;

	// x_k+1 = x_k + a_k * p_k
	for(i=0;i<N;i++){
	  x[i]+=alpha*p[i];
	}
	// r_k+1 = r_k - a_k * A * p_k
	for(i=0;i<N;i++){
	  r[i]+=-alpha*Ap[i];
	}

	// check convergence
	norm = 0.0;
	for(i=0;i<N;i++){
	  norm+=fabs(r[i]);
	}
	printf("norm %e %e\n", norm, eps);
	if(norm < eps)break;

	// b_k = (rt_k+1 * r_k+1) / (rt_k * r_k)
	rr0 = rr;
	rr = 0.0;
	for(i=0;i<N;i++){
	  rr+=r[i]*r[i];
	}
	beta = rr / rr0;
	// p_k+1 = r_k+1 + beta * p_k
	for(i=0;i<N;i++){
	  p[i]=r[i]+beta*p[i];
	}
  }
  t2 = omp_get_wtime();

  printf("time: %f sec , %f sec/iter\n", t2-t1, (t2-t1)/iter);

  // check data
#if 1
  printf("result(x):\n");
  for(i=0;i<N;i++){
	printf(" %f %f\n", x[i], xx[i]);
  }
#endif

  // memory free : omit
  return 0;
}
