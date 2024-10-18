/* Simple Discrete Fourier Transform implemented in C and OpenMP/C */
/* compilation with : gcc -fopenmp -O3 -o MyDFT_OpenMP MyDFT_OpenMP.c -lm -lgomp */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define PI 3.141592653589793

#define MYFLOAT float

void MyDFT(MYFLOAT *A, MYFLOAT *B, MYFLOAT *a, MYFLOAT *b,int size)
{
  for (uint j=0;j<size;j++)
    {
      MYFLOAT At=0.,Bt=0.;
      for (uint i=0; i<size;i++) 
	{
	  At+=a[i]*cos(2.*PI*(MYFLOAT)(j*i)/(MYFLOAT)size)+b[i]*sin(2.*PI*(MYFLOAT)(j*i)/(MYFLOAT)size);
	  Bt+=-a[i]*sin(2.*PI*(MYFLOAT)(j*i)/(MYFLOAT)size)+b[i]*cos(2.*PI*(MYFLOAT)(j*i)/(MYFLOAT)size);
	}
      A[j]=At;
      B[j]=Bt;
    }
}

void MyDFTOpenMP(MYFLOAT *A, MYFLOAT *B, MYFLOAT *a, MYFLOAT *b,int size)
{
  #pragma omp parallel for
  for (uint j=0;j<size;j++)
    {
      MYFLOAT At=0.,Bt=0.;
      for (uint i=0; i<size;i++) 
	{
	  At+=a[i]*cos(2.*PI*(MYFLOAT)(j*i)/(MYFLOAT)size)+b[i]*sin(2.*PI*(MYFLOAT)(j*i)/(MYFLOAT)size);
	  Bt+=-a[i]*sin(2.*PI*(MYFLOAT)(j*i)/(MYFLOAT)size)+b[i]*cos(2.*PI*(MYFLOAT)(j*i)/(MYFLOAT)size);
	}
      A[j]=At;
      B[j]=Bt;
    }
}

int main(int argc,char *argv[])
{
  float *a,*b,*A,*B;
  int size=1024;
  struct timeval tv1,tv2;
 
  if (argc > 1) {
    size=(int)atoll(argv[1]);
  }
  else {
    printf("\n\tEstimate DFT\n\n\t\t#1 : size (default 1024)\n\n");
  }

  a=(float*)malloc(size*sizeof(MYFLOAT));
  b=(float*)malloc(size*sizeof(MYFLOAT));
  A=(float*)malloc(size*sizeof(MYFLOAT));
  B=(float*)malloc(size*sizeof(MYFLOAT));

  for (int i=0;i<size;i++)
    {
      a[i]=1.;
      b[i]=1.;
      A[i]=0.;
      B[i]=0.;
    }

  gettimeofday(&tv1, NULL);
  MyDFTOpenMP(A,B,a,b,size);
  gettimeofday(&tv2, NULL);

  MYFLOAT elapsedOMP=(MYFLOAT)((tv2.tv_sec-tv1.tv_sec) * 1000000L +
			    (tv2.tv_usec-tv1.tv_usec))/1000000;

  /* printf("A=["); */
  /* for (int i=0;i<size;i++) */
  /*   { */
  /*     printf("%.2f ",A[i]); */
  /*   } */
  /* printf(" ]\n\n"); */

  /* printf("B=["); */
  /* for (int i=0;i<size;i++) */
  /*   { */
  /*     printf("%.2f ",B[i]); */
  /*   } */
  /* printf(" ]\n\n"); */

  printf("\nA[0]=%.3f A[%i]=%.3f\n",A[0],size-1,A[size-1]);
  printf("B[0]=%.3f B[%i]=%.3f\n\n",B[0],size-1,B[size-1]);

  printf("OpenMPElapsed: %.3f\n",elapsedOMP);
  printf("OpenMPRate: %.i\n",(int)((float)size/elapsedOMP));
  
  free(a);
  free(b);
  free(A);
  free(B);
}
  
