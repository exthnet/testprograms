#ifndef _TIME_CPU_H
#define _TIME_CPU_H

#include <stdio.h>
#include <string.h>
#include <sys/time.h>           // timezoneに必要

struct SCpuTime{
  struct timeval tBegin, tEnd;
  struct timezone tz;
  double dSec;
  double dMinSec;
  double dMaxSec;
  double dFlops;
  int nFlops;
  int nActive;
  char name[32];
};
#ifdef __cplusplus
extern "C" {
#endif
int TimeInitialize(struct SCpuTime *t, const char *str);
int TimeBegin(struct SCpuTime *t);
int TimeEnd1(struct SCpuTime *t);
int TimeEnd2(struct SCpuTime *t, double flop);
int TimePrintf(struct SCpuTime *t);
#ifdef __cplusplus
}
#endif

#endif
