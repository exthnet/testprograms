/* -*- C++ -*- */
#include "time_cpu_struct.h"

int TimeInitialize(struct SCpuTime *t, const char *str){
  t->dSec = 0.0;
  t->dFlops = 0.0;
  t->nActive = 0;
  return 0;
}
int TimeBegin(struct SCpuTime *t){
  if(t->nActive == 0){
#if WIN32
    QueryPerformanceFrequency(&t->liFreq);
    QueryPerformanceCounter(&t->liBegin);
#else
    gettimeofday(&t->tBegin, &t->tz);
#endif
    t->nActive = 1;
  }else{
    printf("*** already activated\n");
  }
  return 0;
}
int TimeEnd1(struct SCpuTime *t){
  if(t->nActive == 1){
    double sec;
#if WIN32
    QueryPerformanceCounter(&t->liEnd);
    sec = (double)(t->liEnd.QuadPart - t->liBegin.QuadPart)/(double)t->liFreq.QuadPart;
#else
    double dBegin, dEnd;
    gettimeofday(&t->tEnd, &t->tz);
    dBegin= t->tBegin.tv_sec + (double)t->tBegin.tv_usec*1.0e-6;
    dEnd= t->tEnd.tv_sec + (double)t->tEnd.tv_usec*1.0e-6;
    sec = dEnd - dBegin;
#endif
    t->dSec+= sec;
    t->nFlops = 0;
    t->nActive = 0;
  }else{
    printf("*** not activated yet\n");
  }
  return 0;
}
int TimeEnd2(struct SCpuTime *t, double flop){
  if(t->nActive == 1){
    double sec;
#if WIN32
    QueryPerformanceCounter(&t->liEnd);
    sec = (double)(t->liEnd.QuadPart - t->liBegin.QuadPart)/(double)t->liFreq.QuadPart;
#else
    double dBegin, dEnd;
    gettimeofday(&t->tEnd, &t->tz);
    dBegin= t->tBegin.tv_sec + (double)t->tBegin.tv_usec*1.0e-6;
    dEnd= t->tEnd.tv_sec + (double)t->tEnd.tv_usec*1.0e-6;
    sec = dEnd - dBegin;
#endif
    t->dSec+= sec;
    double flops = flop/sec;
    t->dFlops = (t->dFlops+flops)/2.0;
    t->nFlops = 1;
    t->nActive = 0;
  }else{
    printf("*** not activated yet\n");
  }
  return 0;
}
int TimePrintf(struct SCpuTime *t){
  if(t->nActive == 0){
    if(t->nFlops==0){
      printf("%f sec\n", t->dSec);
    }else{
      printf("%f sec, %f Gflops\n", t->dSec, t->dFlops);
    }
  }else{
    printf("*** still activated\n");
  }
  return 0;
}
