// Naive Discrete Fourier Transform in SyCL 
// Emmanuel QUEMENER <emmanuel.quemener@gmail.com>
// To create and activate compete SyCL environment, Debian Bookworm
// Create...
// export DPCPP_HOME=<MyRootInstall>/sycl_workspace
// mv $DPCPP_HOME ${DPCPP_HOME}-$(date "+%Y%m%d-%H%M")
// mkdir $DPCPP_HOME
// cd $DPCPP_HOME
// git clone https://github.com/intel/llvm -b sycl
// python3 $DPCPP_HOME/llvm/buildbot/configure.py --cuda
// python3 $DPCPP_HOME/llvm/buildbot/compile.py
// python3 $DPCPP_HOME/llvm/buildbot/check.py
// Use...
// export DPCPP_HOME=<MyRootInstall>/sycl_workspace
// export PATH=$DPCPP_HOME/llvm/build/bin:$PATH
// export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib:$LD_LIBRARY_PATH
// clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda MyDFT_SyCL.cpp -o MyDFT_SyCL
// ./MyDFT_SyCL 1024

#include <iostream>
#include <sycl/sycl.hpp>
#include <sys/time.h>

#define PI 3.141592653589793

#define MYFLOAT float

using namespace sycl;
void MyDFTSyCL(MYFLOAT *A, MYFLOAT *B, MYFLOAT *a, MYFLOAT *b,int size)
{
  sycl::buffer<MYFLOAT> aBuf(&a[0],size);
  sycl::buffer<MYFLOAT> bBuf(&b[0],size);
  sycl::buffer<MYFLOAT> ABuf(&A[0],size);
  sycl::buffer<MYFLOAT> BBuf(&B[0],size);

  // Creating SYCL queue
  sycl::queue Queue;

  Queue.submit([&](auto &h) {
    // Getting write only access to the buffer on a device.
    sycl::accessor Aa{aBuf, h, sycl::read_only};
    sycl::accessor Ab{bBuf, h, sycl::read_only};
    sycl::accessor AA{ABuf, h};
    sycl::accessor AB{BBuf, h};

    // Executing kernel
    h.parallel_for(size,[=](auto j) {      
      MYFLOAT At=0.,Bt=0.;
      for (uint i=0; i<size;i++) 
        {
          At+=Aa[i]*cos(2.*PI*(MYFLOAT)(j*i)/(MYFLOAT)size)+Ab[i]*sin(2.*PI*(MYFLOAT)(j*i)/(MYFLOAT)size);
          Bt+=-Aa[i]*sin(2.*PI*(MYFLOAT)(j*i)/(MYFLOAT)size)+Ab[i]*cos(2.*PI*(MYFLOAT)(j*i)/(MYFLOAT)size);
        }
      AA[j]=At;
      AB[j]=Bt;
    });
  });

}
  
using namespace sycl;
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

  // all@1 sets size@0  
  // for (int i=0;i<size;i++)
  //   {
  //     a[i]=1.;
  //     b[i]=1.;
  //     A[i]=0.;
  //     B[i]=0.;
  //   }

  // 1@0 sets all@1
  for (int i=0;i<size;i++)
    {
      a[i]=0;
      b[i]=0;
      A[i]=0.;
      B[i]=0.;
    }
  a[0]=(float)size;
  b[0]=(float)size;

  gettimeofday(&tv1, NULL);

  MyDFTSyCL(A,B,a,b,size);
  gettimeofday(&tv2, NULL);

  MYFLOAT elapsedSyCL=(MYFLOAT)((tv2.tv_sec-tv1.tv_sec) * 1000000L +
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

  std::cout << "ElapsedSyCL: " << elapsedSyCL << ::std::endl;
  std::cout << "SyCLRate: " << size/elapsedSyCL << std::endl;
  
  free(a);
  free(b);
  free(A);
  free(B);
}
  
