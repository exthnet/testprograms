// -*- C++ -*-
#include "time_gpu_struct.h"

// timeGetTime
extern "C"
int TimeInitialize_GPU(struct SGpuTime *t, const char *str){
  t->dSec = 0.0;
  t->dMinSec = 9999999.0;
  t->dMaxSec = 0.0;
  t->dFlops = 0.0;
  t->nFlops = 0;
  t->nActive = 0;
  t->nCount = 0;
  strcpy(t->name, str);
  return 0;
}
int TimeBegin_GPU(struct SGpuTime *t){
  if(t->nActive == 0){
    gettimeofday(&t->tBegin, &t->tz);
    t->nActive = 1;
  }else{
    printf("%s *** already activated\n", t->name);
    return 1;
  }
  return 0;
}
int TimeEnd1_GPU(struct SGpuTime *t){
  if(t->nActive == 1){
    double sec;
    cudaError _err = cudaGetLastError();
    if(cudaSuccess!=_err){
      printf("%s error: time_gpu_end %s\n", t->name, cudaGetErrorString(_err));
      return 1;
    }
    _err=cudaDeviceSynchronize();
    if(cudaSuccess!=_err){
      printf("%s error: time_gpu_end %s\n", t->name, cudaGetErrorString(_err));
      return 1;
    }
    double dBegin, dEnd;
    gettimeofday(&t->tEnd, &t->tz);
    dBegin= t->tBegin.tv_sec + (double)t->tBegin.tv_usec*1.0e-6;
    dEnd= t->tEnd.tv_sec + (double)t->tEnd.tv_usec*1.0e-6;
    sec = dEnd - dBegin;
    t->dSec+= sec;
    if(sec > t->dMaxSec)t->dMaxSec = sec;
    if(sec < t->dMinSec)t->dMinSec = sec;
	t->nCount++;
    t->nFlops = 0;
    t->nActive = 0;
  }else{
    printf("%s *** not activated yet\n", t->name);
    return 2;
  }
  return 0;
}
int TimeEnd2_GPU(struct SGpuTime *t, double flop){
  if(t->nActive == 1){
    double sec;
    cudaError _err = cudaGetLastError();
    if(cudaSuccess!=_err){
      printf("%s error: time_gpu_end %s\n", t->name, cudaGetErrorString(_err));
      return 1;
    }
    _err=cudaDeviceSynchronize();
    if(cudaSuccess!=_err){
      printf("%s error: time_gpu_end %s\n", t->name, cudaGetErrorString(_err));
      return 1;
    }
    double dBegin, dEnd;
    gettimeofday(&t->tEnd, &t->tz);
    dBegin= t->tBegin.tv_sec + (double)t->tBegin.tv_usec*1.0e-6;
    dEnd= t->tEnd.tv_sec + (double)t->tEnd.tv_usec*1.0e-6;
    sec = dEnd - dBegin;
    t->dSec+= sec;
    if(sec > t->dMaxSec)t->dMaxSec = sec;
    if(sec < t->dMinSec)t->dMinSec = sec;
	t->nCount++;
    double flops = flop/sec;
	if(t->nCount==1){
	  t->dFlops = flops;
	}else{
	  t->dFlops = (t->dFlops+flops)/2.0;
	}
    t->nFlops = 1;
    t->nActive = 0;
    return 0;
  }else{
    printf("%s *** not activated yet\n", t->name);
    return 2;
  }
}
int TimePrintf_GPU(struct SGpuTime *t){
  if(t->nActive == 0){
    if(t->nFlops==0){
      printf("%s %f sec ( %f - %f sec ) / %d times\n", t->name, t->dSec, t->dMinSec, t->dMaxSec, t->nCount);
    }else{
      printf("%s %f sec, %f Gflops ( %f - %f sec ) / %d times\n", t->name, t->dSec, t->dFlops, t->dMinSec, t->dMaxSec, t->nCount);
    }
    return 0;
  }else{
    printf("%s %f sec ( %f - %f sec )\n", t->name, 0.0, 0.0, 0.0);
    //printf("%s *** still activated\n", t->name);
    return 2;
  }
}
