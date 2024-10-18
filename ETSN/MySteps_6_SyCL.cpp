// Simple addition of two vectors where Mylq function is applied 
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
// clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -DNOCHECK MySteps_6_SyCL.cpp -o MySteps_6_SyCL
// ./MySteps_6_SyCL 1024 10

#include <iostream>
#include <sycl/sycl.hpp>
#include <sys/time.h>

#define MYFLOAT float

#define ERROR 1e-5

using namespace sycl;
MYFLOAT MySillyFunction(MYFLOAT x)
{
    return(pow(sqrt(log(exp(atanh(tanh(asinh(sinh(acosh(cosh(atan(tan(asin(sin(acos(cos(x))))))))))))))),2));
}

using namespace sycl;
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

using namespace sycl;
int main(int argc, char *argv[]) {
  MYFLOAT *a,*b,*res,*resacc;
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

  std::cout << size << " " << calls << std::endl ;

  a=(MYFLOAT*)malloc(size*sizeof(MYFLOAT));
  b=(MYFLOAT*)malloc(size*sizeof(MYFLOAT));
  res=(MYFLOAT*)malloc(size*sizeof(MYFLOAT));
  resacc=(MYFLOAT*)malloc(size*sizeof(MYFLOAT));

  srand(110271);
  // Initialize the vectors
  for (size_t I = 0; I < size; ++I) {
    a[I]=(MYFLOAT)rand()/(MYFLOAT)RAND_MAX;
    b[I]=(MYFLOAT)rand()/(MYFLOAT)RAND_MAX;
    res[I]=0;
  }

#ifdef CHECK
  gettimeofday(&tv1, NULL);
  MySillySum(res,a,b,calls,size);
  gettimeofday(&tv2, NULL);
  MYFLOAT elapsedNative=(MYFLOAT)((tv2.tv_sec-tv1.tv_sec) * 1000000L +
                                  (tv2.tv_usec-tv1.tv_usec))/1000000;

  std::cout << "ElapsedNative: " << elapsedNative << std::endl;
  std::cout << "NativeRate: " << size/elapsedNative << std::endl;
#endif
  
  gettimeofday(&tv1, NULL);
  
  sycl::buffer<MYFLOAT> aBuf(&a[0],size);
  sycl::buffer<MYFLOAT> bBuf(&b[0],size);
  sycl::buffer<MYFLOAT> resaccBuf(&resacc[0],size);

  // Creating SYCL queue
  sycl::queue Queue;

  Queue.submit([&](auto &h) {
    // Getting write only access to the buffer on a device.
    sycl::accessor Aa{aBuf, h, sycl::read_only};
    sycl::accessor Ab{bBuf, h, sycl::read_only};
    sycl::accessor Aresacc{resaccBuf, h};
    
    // Executing kernel
    h.parallel_for(size,[=](auto i) {      
      MYFLOAT Aai = Aa[i];
      MYFLOAT Abi = Ab[i] ;
      for (size_t C = 0 ; C < calls ; C++) {
        Aai = MySillyFunction(Aai);
        Abi = MySillyFunction(Abi) ;
      }
      Aresacc[i] = Aai + Abi ;
    });
  });

  // Getting read only access to the buffer on the host.
  // Implicit barrier waiting for queue to complete the work.
  sycl::host_accessor HostAccessor{resaccBuf};

  gettimeofday(&tv2, NULL);

  MYFLOAT elapsedSyCL=(MYFLOAT)((tv2.tv_sec-tv1.tv_sec) * 1000000L +
                                  (tv2.tv_usec-tv1.tv_usec))/1000000;

  // Check the results
#ifdef CHECK
  bool MismatchFound = false;
  for (size_t I = 0; I < size; ++I) {
    if ( fabs(resacc[I] - res[I]) > ERROR ) {
      std::cout << "Element: " << I << ", error: " << res[I]-resacc[I]
                << std::endl;
      MismatchFound = true;
    }
  }

  if (!MismatchFound) {
    std::cout << "The results are correct!" << std::endl;
  }
  // return MismatchFound;
#endif
 
  std::cout << "ElapsedSyCL: " << elapsedSyCL << ::std::endl;
  std::cout << "SyCLRate: " << size/elapsedSyCL << std::endl;
  
  free(a);
  free(b);
  free(res);
  free(resacc);

}
