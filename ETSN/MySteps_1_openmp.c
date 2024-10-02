/* Simple Sum function in C and OpenMP/C */
/* compilation with sequential compute : gcc -fopenmp -O3 -o MySteps_1_openmp MySteps_1_openmp.c -lm -lgomp */
/* compilation without sequential compute : gcc -DNOSERIAL -fopenmp -O3 -o MySteps_1_openmp_1_NoSerial MySteps_1_openmp.c -lm -lgomp */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define PI 3.141592653589793

#define MYFLOAT float

void MySum(MYFLOAT *res, MYFLOAT *a, MYFLOAT *b, int size)
{
  for (uint i=0; i<size;i++) 
    {
      res[i] = a[i] + b[i];
    }
}

void MySillySumOMP(MYFLOAT *res, MYFLOAT *a, MYFLOAT *b, int size)
{
  #pragma omp parallel for
  for (uint i=0; i<size;i++) 
    {
      res[i] = a[i] + b[i];
    }
}

MYFLOAT MyNorm(MYFLOAT *a,MYFLOAT *b,int size)
{
  MYFLOAT norm=0.;

  for (int i=0;i<size;i++)
    {
      norm+=pow(a[i]-b[i],2);
    }

  return(sqrt(norm));
}

void MyPrint(MYFLOAT *a,int size)
{
  printf("[");
  for (int i=0;i<size;i++)
    {
      printf(" %.8e ",a[i]);
    }
  printf("]\n");
}

int main(int argc,char *argv[])
{
  float *a,*b,*res,*resacc;
  int size=1024;
  struct timeval tv1,tv2;
 
  if (argc > 1) {
    size=(int)atoll(argv[1]);
  }
  else {
    printf("\n\tPi : Estimate SillySum\n\n\t\t#1 : size (default 1024)\n\n");
  }

  printf("%i\n",size);
  
  a=(float*)malloc(size*sizeof(MYFLOAT));
  b=(float*)malloc(size*sizeof(MYFLOAT));
  res=(float*)malloc(size*sizeof(MYFLOAT));
  resacc=(float*)malloc(size*sizeof(MYFLOAT));

  srand(110271);
  
  for (int i=0;i<size;i++)
    {
      a[i]=(MYFLOAT)rand()/(MYFLOAT)RAND_MAX;
      b[i]=(MYFLOAT)rand()/(MYFLOAT)RAND_MAX;
      res[i]=0.;
      resacc[i]=0.;
    }

#ifndef NOSERIAL 
  gettimeofday(&tv1, NULL);
  MySum(res,a,b,size);
  gettimeofday(&tv2, NULL);
#endif
  
  MYFLOAT elapsed=(MYFLOAT)((tv2.tv_sec-tv1.tv_sec) * 1000000L +
			    (tv2.tv_usec-tv1.tv_usec))/1000000;

  gettimeofday(&tv1, NULL);
  MySillySumOMP(resacc,a,b,size);
  gettimeofday(&tv2, NULL);

  MYFLOAT elapsedAcc=(MYFLOAT)((tv2.tv_sec-tv1.tv_sec) * 1000000L +
			       (tv2.tv_usec-tv1.tv_usec))/1000000;

#ifndef NOSERIAL 
  MYFLOAT MyChecker=MyNorm(res,resacc,size);
  printf("Norm: %.8e\n",MyChecker);
#endif

#ifdef VERBOSE
  MyPrint(res,size);
  MyPrint(resacc,size);
#endif
  
#ifndef NOSERIAL 
  printf("Elapsed Time: %.3f\n",elapsed);
#endif
  
  printf("OMP Elapsed Time: %.3f\n",elapsedAcc);

#ifndef NOSERIAL 
  printf("NaiveRate: %.lld\n",(unsigned long)((float)size/elapsed));
#endif
  printf("OMPRate: %.lld\n",(unsigned long)((float)size/elapsedAcc));

#ifndef NOSERIAL   
  printf("AccRatio: %.3f\n",elapsed/elapsedAcc);
#endif
 
  free(a);
  free(b);
  free(res);
  free(resacc);
}
  
