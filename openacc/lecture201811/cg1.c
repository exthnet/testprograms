#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Congugate Gradient Method: solve Ax=b

double frand(double d)
{
  return (double)((rand()+1)%100)/100.0 * d;
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
  double bnrm, dnrm, resid, rho, rho1, pq, *z, *q, *dd;
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
  double x[10];
  double xx[10] = {1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0};
#else
  if(argc>1){N=atoi(argv[1]); printf("N=%d\n",N);}
  if(argc>2)randseed=atoi(argv[1]);
  if(argc>3)randseed=atoi(argv[2]);

  double *A, *b, *x, *xx;
  srand(randseed);
  A = (double*)malloc(sizeof(double)*N*N);
  b = (double*)malloc(sizeof(double)*N);
  x = (double*)malloc(sizeof(double)*N);
  xx = (double*)malloc(sizeof(double)*N);
  /*
	for(i=0;i<N;i++){
	for(j=i;j<N;j++){
	double d = frand(100.0);
	A[i*N+j]=d;
	A[j*N+i]=d;
	}
	}
  */
  for(i=0;i<N;i++){
	for(j=0;j<N;j++){
	  A[i*N+j]=-1.0;
	}
  }
  for(i=0;i<N;i++){
	A[i*N+i]=(double)(N-1);
  }

  for(i=0;i<N;i++){
	x[i]=0.0;
	xx[i]=frand(100.0);
  }
  for(i=0;i<N;i++){
	b[i] = 0.0;
	for(j=0;j<N;j++){
	  b[i]+=A[i*N+j]*xx[j];
	}
  }
  for(i=0;i<N;i++)x[i]=0.0;
#endif
  r = (double*)malloc(sizeof(double)*N);
  p = (double*)malloc(sizeof(double)*N);
  Ap = (double*)malloc(sizeof(double)*N);
  Ax = (double*)malloc(sizeof(double)*N);
  z = (double*)malloc(sizeof(double)*N);
  q = (double*)malloc(sizeof(double)*N);
  dd = (double*)malloc(sizeof(double)*N);

  // check data
#if 0
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

  // Initialize
  // {x}=0.0
  for(i=0;i<N;i++){
	x[i]=0.0;
	r[i]=0.0;
	z[i]=0.0;
	q[i]=0.0;
	p[i]=0.0;
	dd[i]=0.0;
  }

  // {r0} = {b} - [A]{xini}
  for(i=0;i<N;i++){
	Ax[i] = 0.0;
	for(j=0;j<N;j++){
	  Ax[i] += A[i*N+j]*x[j];
	}
  }
  for(i=0;i<N;i++){
	r[i]=b[i]-Ax[i];
  }
  for(i=0;i<N;i++){
	dd[i] = 1.0/A[i*N+i];
  }

  bnrm = 0.0;
  for(i=0;i<N;i++){
	bnrm += b[i]*b[i];
  }
  if(bnrm==0.0)bnrm=1.0;

  t1 = omp_get_wtime();
  for(iter=1; iter<=maxiter; iter++){
	printf("iter %d ", iter);

	// {z} = [Minv]{r}
	for(i=0;i<N;i++){
	  z[i] = dd[i]*r[i];
	}

	// {rho} = {r}{z}
	rho = 0.0;
	for(i=0;i<N;i++){
	  rho += r[i]*z[i];
	}

	// {p} = {z} if iter=1
	// beta = rho/rho1 otherwise
	if(iter==1){
	  for(i=0;i<N;i++){
		p[i] = z[i];
	  }
	}else{
	  beta = rho/rho1;
	  for(i=0;i<N;i++){
		p[i] = z[i] + beta*p[i];
	  }
	}

	// {q} = [A]{p}
	for(i=0;i<N;i++){
	  q[i] = 0.0;
	  for(j=0;j<N;j++){
		q[i] += A[i*N+j]*p[j];
	  }
	}

	// alpha = rho / {p}{q}
	pq = 0.0;
	for(i=0;i<N;i++){
	  pq += p[i]*q[i];
	}
	alpha = rho / pq;

	// {x} = {x} + alpha*{p}
	// {r} = {r} - alpha*{q}
	for(i=0;i<N;i++){
	  x[i] += + alpha*p[i];
	  r[i] += - alpha*q[i];
	}

	// check converged
	dnrm = 0.0;
	for(i=0;i<N;i++){
	  dnrm += r[i]*r[i];
	}
	resid = sqrt(dnrm/bnrm);
	printf(" %e %e\n", resid, eps);
	if(resid <= eps){break;}
	if(iter == maxiter){break;}
	rho1 = rho;
  }
  t2 = omp_get_wtime();

  printf("time: %f sec , %f sec/iter\n", t2-t1, (t2-t1)/iter);

  // check data
#if 0
  printf("result(x):\n");
  for(i=0;i<N;i++){
	printf(" %f\n", x[i]);
  }
#endif
  for(i=0;i<N;i++){
  printf("%d %f %f: %e\n", i+1, xx[i], x[i], xx[i]-x[i]);
  }

  // memory free : omit
  return 0;
}
