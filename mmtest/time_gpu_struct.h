#ifndef _TIME_GPU_H
#define _TIME_GPU_H

#include <stdio.h>
//#include <stdlib.h>
//#include <stdarg.h>
#include <string.h>
#include <sys/time.h>           // timezoneに必要
//#include <cuda_runtime_api.h>

struct SGpuTime{
  struct timeval tBegin, tEnd;
  struct timezone tz;
  double dSec;
  double dMinSec;
  double dMaxSec;
  double dFlops;
  int nFlops;
  int nActive;
  char name[32];
  int nCount;
};
#ifdef __cplusplus
extern "C" {
#endif
int TimeInitialize_GPU(struct SGpuTime *t, const char *str);
int TimeBegin_GPU(struct SGpuTime *t);
int TimeEnd1_GPU(struct SGpuTime *t);
int TimeEnd2_GPU(struct SGpuTime *t, double flop);
int TimePrintf_GPU(struct SGpuTime *t);
#ifdef __cplusplus
}
#endif

#endif // _TIME_GPU_H
