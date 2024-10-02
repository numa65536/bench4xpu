/* Simple SillySum function in C */
/* compilation with : gcc -O3 -o MySteps_6 MySteps_6.c -lm */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define PI 3.141592653589793

#define MYFLOAT float

MYFLOAT MySillyFunction(MYFLOAT x)
{
    return(pow(sqrt(log(exp(atanh(tanh(asinh(sinh(acosh(cosh(atan(tan(asin(sin(acos(cos(x))))))))))))))),2)); 
}

void MySillySum(MYFLOAT *res, MYFLOAT *a, MYFLOAT *b,int calls, int size)
{
  for (uint i=0; i<size;i++) 
    {
      MYFLOAT ai=a[i];
      MYFLOAT bi=b[i];
      
      for (int c=0;c<calls;c++)
	{
	  ai=MySillyFunction(ai);
	  bi=MySillyFunction(bi);
	}

      res[i] = ai + bi;
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
  int calls=1;
  struct timeval tv1,tv2;
 
  if (argc > 1) {
    size=(int)atoll(argv[1]);
    calls=(int)atoll(argv[2]);
  }
  else {
    printf("\n\tMySteps : Estimate SillySum\n\n\t\t#1 : size (default 1024)\n\t\t#2 : calls (default 1)\n\n");
  }

  printf("%i %i\n",size,calls);
  
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

  gettimeofday(&tv1, NULL);
  MySillySum(res,a,b,calls,size);
  gettimeofday(&tv2, NULL);

  MYFLOAT elapsed=(MYFLOAT)((tv2.tv_sec-tv1.tv_sec) * 1000000L +
			    (tv2.tv_usec-tv1.tv_usec))/1000000;

#ifdef VERBOSE
  MyPrint(res,size);
  MyPrint(resacc,size);
#endif
  
  printf("Elapsed Time: %.3f\n",elapsed);

  printf("NativeRate: %.lld\n",(unsigned long)((float)size/elapsed));

  free(a);
  free(b);
  free(res);
  free(resacc);
}
  
