#include "time_cpu_struct.h"


#if 1
// timeGetTime

int TimeInitialize(struct SCpuTime *t, const char *str){
  t->dSec = 0.0;
  t->dMinSec = 9999999.0;
  t->dMaxSec = 0.0;
  t->dFlops = 0.0;
  t->nFlops = 0;
  t->nActive = 0;
	strcpy(t->name, str);
  return 0;
}
int TimeBegin(struct SCpuTime *t){
  if(t->nActive == 0){
#ifdef _WIN32
		t->dwBegin = timeGetTime();
#endif
#ifdef _LINUX
    gettimeofday(&t->tBegin, &t->tz);
#endif
    t->nActive = 1;
    return 0;
  }else{
    printf("%s *** already activated\n", t->name);
    return -1;
  }
}
int TimeEnd1(struct SCpuTime *t){
  if(t->nActive == 1){
    double sec;
#ifdef _WIN32
		t->dwEnd = timeGetTime();
    sec = (t->dwEnd-t->dwBegin)/1000.0;
#endif
#ifdef _LINUX
    double dBegin, dEnd;
    gettimeofday(&t->tEnd, &t->tz);
    dBegin= t->tBegin.tv_sec + (double)t->tBegin.tv_usec*1.0e-6;
    dEnd= t->tEnd.tv_sec + (double)t->tEnd.tv_usec*1.0e-6;
    sec = dEnd - dBegin;
#endif
    t->dSec+= sec;
    if(sec > t->dMaxSec)t->dMaxSec = sec;
    if(sec < t->dMinSec)t->dMinSec = sec;
    t->nFlops = 0;
    t->nActive = 0;
    return 0;
  }else{
    printf("%s *** not activated yet\n", t->name);
    return -1;
  }
}
int TimeEnd2(struct SCpuTime *t, double flop){
  if(t->nActive == 1){
    double sec;
#ifdef _WIN32
		t->dwEnd = timeGetTime();
    sec = (t->dwEnd-t->dwBegin)/1000.0;
#endif
#ifdef _LINUX
    double dBegin, dEnd;
    gettimeofday(&t->tEnd, &t->tz);
    dBegin= t->tBegin.tv_sec + (double)t->tBegin.tv_usec*1.0e-6;
    dEnd= t->tEnd.tv_sec + (double)t->tEnd.tv_usec*1.0e-6;
    sec = dEnd - dBegin;
#endif
    t->dSec+= sec;
    if(sec > t->dMaxSec)t->dMaxSec = sec;
    if(sec < t->dMinSec)t->dMinSec = sec;
    double flops = flop/sec;
    t->dFlops = (t->dFlops+flops)/2.0;
    t->nFlops = 1;
    t->nActive = 0;
    return 0;
  }else{
    printf("%s *** not activated yet\n", t->name);
    return -1;
  }
}
int TimePrintf(struct SCpuTime *t){
  if(t->nActive == 0){
    if(t->nFlops==0){
      printf("%s%f sec ( %f -%f sec )\n", t->name, t->dSec, t->dMinSec, t->dMaxSec);
    }else{
      printf("%s%f sec, %f Gflops ( %f -%f sec )\n", t->name, t->dSec, t->dFlops, t->dMinSec, t->dMaxSec);
    }
    return 0;
  }else{
    printf("%s *** still activated\n", t->name);
    return -1;
  }
}


#else
// QueryPerformance

int TimeInitialize(struct SCpuTime *t, const char *str){
  t->dSec = 0.0;
  t->dFlops = 0.0;
  t->nFlops = 0;
  t->nActive = 0;
	strcpy(t->name, str);
  return 0;
}
int TimeBegin(struct SCpuTime *t){
  if(t->nActive == 0){
#ifdef _WIN32
    QueryPerformanceFrequency(&t->liFreq);
    QueryPerformanceCounter(&t->liBegin);
#else
    gettimeofday(&t->tBegin, &t->tz);
#endif
    t->nActive = 1;
    return 0;
  }else{
    printf("%s*** already activated\n", t->name);
    return -1;
  }
}
int TimeEnd1(struct SCpuTime *t){
  if(t->nActive == 1){
    double sec;
#ifdef _WIN32
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
    return 0;
  }else{
    printf("%s*** not activated yet\n", t->name);
    return -1;
  }
}
int TimeEnd2(struct SCpuTime *t, double flop){
  if(t->nActive == 1){
    double sec;
#ifdef _WIN32
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
    return 0;
  }else{
    printf("%s*** not activated yet\n", t->name);
    return -1;
  }
}
int TimePrintf(struct SCpuTime *t){
  if(t->nActive == 0){
    if(t->nFlops==0){
      printf("%s%f sec\n", t->name, t->dSec);
    }else{
      printf("%s%f sec, %f Gflops\n", t->name, t->dSec, t->dFlops);
    }
    return 0;
  }else{
    printf("%s*** still activated\n" t->name);
    return -1;
  }
}

#endif
