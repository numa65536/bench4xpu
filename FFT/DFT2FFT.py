#!/usr/bin/env python3

import numpy as np
import pyopencl as cl
from numpy import pi,cos,sin

#
def NumpyFFT(x,y):
    xy=np.csingle(x+1.j*y)
    XY=np.fft.fft(xy)
    return(XY.real,XY.imag)

#
def GPyFFT(x,y,device):
    import pyopencl as cl
    import pyopencl.array as cla
    import time
    import gpyfft
    from gpyfft.fft import FFT

    TimeIn=time.time()
    Id=0
    HasXPU=False
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if Id==Device:
                XPU=device
                print("CPU/GPU selected: ",device.name.lstrip())
                HasXPU=True
            Id+=1
            # print(Id)

    if HasXPU==False:
        print("No XPU #%i found in all of %i devices, sorry..." % (Device,Id-1))
        sys.exit()           
    Elapsed=time.time()-TimeIn
    print("Selection of device : %.3f" % Elapsed)

    TimeIn=time.time()
    try:
        ctx = cl.Context(devices=[XPU])
        queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
    except:
        print("Crash during context creation")
    Elapsed=time.time()-TimeIn
    print("Context initialisation : %.3f" % Elapsed)

    TimeIn=time.time()
    XY_gpu = cla.to_device(queue, np.csingle(x+1.j*y))
    Elapsed=time.time()-TimeIn
    print("Copy from Host to Device : %.3f" % Elapsed)

    TimeIn=time.time()
    transform = FFT(ctx, queue, XY_gpu)
    event, = transform.enqueue()
    event.wait()
    Elapsed=time.time()-TimeIn
    print("Compute FFT : %.3f" % Elapsed)
    TimeIn=time.time()
    XY = XY_gpu.get()
    Elapsed=time.time()-TimeIn
    print("Copy from Device to Host : %.3f" % Elapsed)
    return(XY.real,XY.imag)

def OpenCLFFT(x,y,device):
    import pyopencl as cl
    import pyopencl.array as cl_array
    import time

    TimeIn=time.time()
    Id=0
    HasXPU=False
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if Id==Device:
                XPU=device
                print("CPU/GPU selected: ",device.name.lstrip())
                HasXPU=True
            Id+=1
            # print(Id)

    if HasXPU==False:
        print("No XPU #%i found in all of %i devices, sorry..." % (Device,Id-1))
        sys.exit()           
    Elapsed=time.time()-TimeIn
    print("Selection of device : %.3f" % Elapsed)

    TimeIn=time.time()
    try:
        ctx = cl.Context(devices=[XPU])
        queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
    except:
        print("Crash during context creation")
    Elapsed=time.time()-TimeIn
    print("Context initialisation : %.3f" % Elapsed)

    prg = cl.Program(ctx, """
__kernel void bit_reversal(__global float2 *data, int N)
{
    int j = 0;
    for (int i=1;i<N-1;i++)
    {
    int k = N / 2;
    while (j >= k) {
    j -= k;
    k /= 2;
    }
    j += k;
    if (i < j)
    {
    float2 temp=data[i];
    data[i]=data[j];
    data[j]=temp;
    }
    }
}

__kernel void fft(__global float2 *data, int N)
{
    int i = N*get_global_id(0);

    float angle = (float)-2.e0f*(float)3.14159e0f/(float)N;
    float2 wlen = (float2)(cos(angle),sin(angle));
    float2 c = (float2)(1.0f,0.0f);
    
    for (int j = 0; j < N / 2; j++) {
         float2 a = data[i + j];
         float2 b = (float2)(data[i+j+N/2].x*c.x - data[i+j+N/2].y*c.y,
                             data[i+j+N/2].x*c.y + data[i+j+N/2].y*c.x);
                
         data[i+j] = (float2)(a.x + b.x , a.y + b.y);
         data[i+j+N/2] = (float2)(a.x - b.x , a.y - b.y);

         float2 tempW = (float2)(c.x*wlen.x - c.y*wlen.y, c.x*wlen.y + c.y*wlen.x);
         c = tempW;
    }
}
    """).build()

    size=len(x)
    
    # Bit reversal
    TimeIn=time.time()
    BITREVERSAL4GPU=True
    if BITREVERSAL4GPU:
        XY_gpu = cl_array.to_device(queue, np.csingle(x+1.j*y))
        bit_reversal_kernel = prg.bit_reversal
        global_size = (1,)
        local_size = None
        bit_reversal_kernel(queue, global_size, local_size, XY_gpu.data, np.int32(size)).wait()
    else:
        XY=np.csingle(x+1.j*y)
        j = 0
        for i in range(1,size-1):
            k = size // 2;
            while j >= k :
                    j -= k;
                    k //= 2;
            j += k
            if i < j :
                XY[i],XY[j]=XY[j],XY[i]
        print(XY)
        XY_gpu = cl_array.to_device(queue, XY)
    
        
    Elapsed=time.time()-TimeIn
    print("Bit reversal : %.3f" % Elapsed)
    
    # Perform FFT
    TimeIn=time.time()
    fft_kernel = prg.fft
    local_size = None
    for stage in range(int(np.log2(size))):
        MyN=2*2**stage
        global_size = (size//MyN,)
        fft_kernel(queue, global_size, local_size, XY_gpu.data, np.int32(MyN)).wait()

    Elapsed=time.time()-TimeIn
    print("FFT Coley Tukey : %.3f" % Elapsed)

    TimeIn=time.time()
    XY = XY_gpu.get()
    Elapsed=time.time()-TimeIn
    print("Copy from Device to Host : %.3f" % Elapsed)
    return(XY.real,XY.imag)

# Naive Discrete Fourier Transform
def MyDFT(x,y):
    size=x.shape[0]
    X=np.zeros(size).astype(np.float32)
    Y=np.zeros(size).astype(np.float32)
    for i in range(size):
        for j in range(size):
            X[i]=X[i]+x[j]*cos(2.*pi*i*j/size)+y[j]*sin(2.*pi*i*j/size)
            Y[i]=Y[i]-x[j]*sin(2.*pi*i*j/size)+y[j]*cos(2.*pi*i*j/size)
    return(X,Y)

# Numpy Discrete Fourier Transform
def NumpyDFT(x,y):
    size=x.shape[0]
    X=np.zeros(size).astype(np.float32)
    Y=np.zeros(size).astype(np.float32)
    nj=np.multiply(2.0*np.pi/size,np.arange(size)).astype(np.float32)
    for i in range(size):
        X[i]=np.sum(np.add(np.multiply(np.cos(i*nj),x),np.multiply(np.sin(i*nj),y)))
        Y[i]=np.sum(-np.subtract(np.multiply(np.sin(i*nj),x),np.multiply(np.cos(i*nj),y)))
    return(X,Y)

# Numba Discrete Fourier Transform
import numba
@numba.njit(parallel=True)
def NumbaDFT(x,y):
    size=x.shape[0]
    X=np.zeros(size).astype(np.float32)
    Y=np.zeros(size).astype(np.float32)
    nj=np.multiply(2.0*np.pi/size,np.arange(size)).astype(np.float32)
    for i in numba.prange(size):
        X[i]=np.sum(np.add(np.multiply(np.cos(i*nj),x),np.multiply(np.sin(i*nj),y)))
        Y[i]=np.sum(-np.subtract(np.multiply(np.sin(i*nj),x),np.multiply(np.cos(i*nj),y)))
    return(X,Y)

# OpenCL complete operation
def OpenCLDFT(a_np,b_np,Device):

    Id=0
    HasXPU=False
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if Id==Device:
                XPU=device
                print("CPU/GPU selected: ",device.name.lstrip())
                HasXPU=True
            Id+=1
            # print(Id)

    if HasXPU==False:
        print("No XPU #%i found in all of %i devices, sorry..." % (Device,Id-1))
        sys.exit()           

    try:
        ctx = cl.Context(devices=[XPU])
        queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
    except:
        print("Crash during context creation")

    TimeIn=time.time()
    # Copy from Host to Device using pointers
    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
    Elapsed=time.time()-TimeIn
    print("Copy from Host 2 Device : %.3f" % Elapsed)

    TimeIn=time.time()
    # Definition of kernel under OpenCL
    prg = cl.Program(ctx, """

#define PI 3.141592653589793

__kernel void MyDFT(
    __global const float *a_g, __global const float *b_g, __global float *A_g, __global float *B_g)
{
  int gid = get_global_id(0);
  uint size = get_global_size(0);
  float A=0.,B=0.;
  for (uint i=0; i<size;i++) 
  {
     A+=a_g[i]*cos(2.*PI*(float)(gid*i)/(float)size)+b_g[i]*sin(2.*PI*(float)(gid*i)/(float)size);
     B+=-a_g[i]*sin(2.*PI*(float)(gid*i)/(float)size)+b_g[i]*cos(2.*PI*(float)(gid*i)/(float)size);
  }
  A_g[gid]=A;
  B_g[gid]=B;
}
""").build()
    Elapsed=time.time()-TimeIn
    print("Building kernels : %.3f" % Elapsed)
    
    TimeIn=time.time()
    # Memory allocation on Device for result
    A_ocl = np.empty_like(a_np)
    B_ocl = np.empty_like(a_np)
    Elapsed=time.time()-TimeIn
    print("Allocation on Host for results : %.3f" % Elapsed)

    A_g = cl.Buffer(ctx, mf.WRITE_ONLY, A_ocl.nbytes)
    B_g = cl.Buffer(ctx, mf.WRITE_ONLY, B_ocl.nbytes)
    Elapsed=time.time()-TimeIn
    print("Allocation on Device for results : %.3f" % Elapsed)

    TimeIn=time.time()
    # Synthesis of function "sillysum" inside Kernel Sources
    knl = prg.MyDFT  # Use this Kernel object for repeated calls
    Elapsed=time.time()-TimeIn
    print("Synthesis of kernel : %.3f" % Elapsed)

    TimeIn=time.time()
    # Call of kernel previously defined 
    CallCL=knl(queue, a_np.shape, None, a_g, b_g, A_g, B_g)
    # 
    CallCL.wait()
    Elapsed=time.time()-TimeIn
    print("Execution of kernel : %.3f" % Elapsed)

    TimeIn=time.time()
    # Copy from Device to Host
    cl.enqueue_copy(queue, A_ocl, A_g)
    cl.enqueue_copy(queue, B_ocl, B_g)
    Elapsed=time.time()-TimeIn
    print("Copy from Device 2 Host : %.3f" % Elapsed)

    # Liberation of memory
    a_g.release()
    b_g.release()
    A_g.release()
    B_g.release()
    
    return(A_ocl,B_ocl)

# CUDA complete operation
def CUDADFT(a_np,b_np,Device,Threads):
    # import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
    
    try:
        # For PyCUDA import
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        
        cuda.init()
        for Id in range(cuda.Device.count()):
            if Id==Device:
                XPU=cuda.Device(Id)
                print("GPU selected %s" % XPU.name())
        print

    except ImportError:
        print("Platform does not seem to support CUDA")

    Context=XPU.make_context()
        
    TimeIn=time.time()
    mod = SourceModule("""

#define PI 3.141592653589793

__global__ void MyDFT(float *A_g, float *B_g, const float *a_g,const float *b_g)
{
  const int gid = blockIdx.x*blockDim.x+threadIdx.x;
  uint size = gridDim.x*blockDim.x;
  float A=0.,B=0.;
  for (uint i=0; i<size;i++) 
  {
     A+=a_g[i]*cos(2.*PI*(float)(gid*i)/(float)size)+b_g[i]*sin(2.*PI*(float)(gid*i)/(float)size);
     B+=-a_g[i]*sin(2.*PI*(float)(gid*i)/(float)size)+b_g[i]*cos(2.*PI*(float)(gid*i)/(float)size);
  }
  A_g[gid]=A;
  B_g[gid]=B;
}

""")
    Elapsed=time.time()-TimeIn
    print("Definition of kernel : %.3f" % Elapsed)

    TimeIn=time.time()
    MyDFT = mod.get_function("MyDFT")
    Elapsed=time.time()-TimeIn
    print("Synthesis of kernel : %.3f" % Elapsed)

    TimeIn=time.time()
    A_np = np.zeros_like(a_np)
    B_np = np.zeros_like(a_np)
    Elapsed=time.time()-TimeIn
    print("Allocation on Host for results : %.3f" % Elapsed)

    Size=a_np.size
    if (Size % Threads != 0):
        print("Impossible : %i not multiple of %i..." % (Threads,Size) )
        TimeIn=time.time()
        MyDFT(drv.Out(A_np), drv.Out(B_np), drv.In(a_np), drv.In(b_np),
              block=(1,1,1), grid=(a_np.size,1))
        Elapsed=time.time()-TimeIn
        print("Execution of kernel : %.3f" % Elapsed)
    else:
        Blocks=int(Size/Threads)
        TimeIn=time.time()
        MyDFT(drv.Out(A_np), drv.Out(B_np), drv.In(a_np), drv.In(b_np),
              block=(Threads,1,1), grid=(Blocks,1))
        Elapsed=time.time()-TimeIn
        print("Execution of kernel : %.3f" % Elapsed)
        
    Context.pop()
    Context.detach()
    
    return(A_np,B_np)

import sys
import time

if __name__=='__main__':

    SIZE=8
    Device=0
    NaiveMethod=False
    NumpyFFTMethod=True
    GPyFFTMethod=True
    OpenCLFFTMethod=True
    NumpyMethod=False
    NumbaMethod=False
    OpenCLMethod=False
    CUDAMethod=False
    Threads=1
    Verbose=True
    
    import getopt

    HowToUse='%s -n [Naive] -y [numpY] -a [numbA] -o [OpenCL] -c [CUDA] -s <SizeOfVector> -d <DeviceId> -t <threads>'
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"vnyaochs:d:t:",["size=","device="])
    except getopt.GetoptError:
        print(HowToUse % sys.argv[0])
        sys.exit(2)

    # List of Devices
    Devices=[]
    Alu={}
        
    for opt, arg in opts:
        if opt == '-h':
            print(HowToUse % sys.argv[0])

            print("\nInformations about devices detected under OpenCL API:")
            # For PyOpenCL import
            try:
                import pyopencl as cl
                Id=0
                for platform in cl.get_platforms():
                    for device in platform.get_devices():
                        #deviceType=cl.device_type.to_string(device.type)
                        deviceType="xPU"
                        print("Device #%i from %s of type %s : %s" % (Id,platform.vendor.lstrip(),deviceType,device.name.lstrip()))
                        Id=Id+1

            except:
                print("Your platform does not seem to support OpenCL")

            print("\nInformations about devices detected under CUDA API:")
            # For PyCUDA import
            try:
                import pycuda.driver as cuda
                cuda.init()
                for Id in range(cuda.Device.count()):
                    device=cuda.Device(Id)
                    print("Device #%i of type GPU : %s" % (Id,device.name()))
                print
            except:
                print("Your platform does not seem to support CUDA")
        
            sys.exit()
        
        elif opt in ("-d", "--device"):
            Device=int(arg)
        elif opt in ("-s", "--size"):
            SIZE = int(arg)
        elif opt in ("-t", "--threads"):
            Threads = int(arg)
        elif opt in ("-n"):
            NaiveMethod=True
        elif opt in ("-y"):
            NumpyMethod=True
        elif opt in ("-a"):
            NumbaMethod=True
        elif opt in ("-o"):
            OpenCLMethod=True
        elif opt in ("-c"):
            CUDAMethod=True

    print("Device Selection : %i" % Device)
    print("Size of complex vector : %i" % SIZE)
    print("DFT Naive computation %s " % NaiveMethod )
    print("DFT Numpy computation %s " % NumpyMethod )
    print("FFT Numpy computation %s " % NumpyFFTMethod )
    print("DFT Numba computation %s " % NumbaMethod )
    print("DFT OpenCL computation %s " % OpenCLMethod )
    print("FFT GPyFFT computation %s " % GPyFFTMethod )
    print("FFT OpenCL computation %s " % OpenCLFFTMethod )
    print("DFT CUDA computation %s " % CUDAMethod )
    
    if CUDAMethod:
        try:
            # For PyCUDA import
            import pycuda.driver as cuda
            
            cuda.init()
            for Id in range(cuda.Device.count()):
                device=cuda.Device(Id)
                print("Device #%i of type GPU : %s" % (Id,device.name()))
                if Id in Devices:
                    Alu[Id]='GPU'
            
        except ImportError:
            print("Platform does not seem to support CUDA")

    if OpenCLMethod:
        try:
            # For PyOpenCL import
            import pyopencl as cl
            Id=0
            for platform in cl.get_platforms():
                for device in platform.get_devices():
                    #deviceType=cl.device_type.to_string(device.type)
                    deviceType="xPU"
                    print("Device #%i from %s of type %s : %s" % (Id,platform.vendor.lstrip().rstrip(),deviceType,device.name.lstrip().rstrip()))

                    if Id in Devices:
                    # Set the Alu as detected Device Type
                        Alu[Id]=deviceType
                    Id=Id+1
        except ImportError:
            print("Platform does not seem to support OpenCL")

    
        
    a_np = np.ones(SIZE).astype(np.float32)
    b_np = np.ones(SIZE).astype(np.float32)
    # a_np = np.random.rand(SIZE).astype(np.float32)
    # b_np = np.random.rand(SIZE).astype(np.float32)

    C_np = np.zeros(SIZE).astype(np.float32)
    D_np = np.zeros(SIZE).astype(np.float32)
    C_np[0] = np.float32(SIZE)
    D_np[0] = np.float32(SIZE)
    
    # Native & Naive Implementation
    if NaiveMethod:
        print("Performing naive implementation")
        TimeIn=time.time()
        c_np,d_np=MyDFT(a_np,b_np)
        NativeElapsed=time.time()-TimeIn
        NativeRate=int(SIZE/NativeElapsed)
        print("NativeRate: %i" % NativeRate)
        print("Precision: ",np.linalg.norm(c_np-C_np),
              np.linalg.norm(d_np-D_np)) 

    # Native & Numpy Implementation
    if NumpyMethod:
        print("Performing Numpy implementation")
        TimeIn=time.time()
        e_np,f_np=NumpyDFT(a_np,b_np)
        NumpyElapsed=time.time()-TimeIn
        NumpyRate=int(SIZE/NumpyElapsed)
        print("NumpyRate: %i" % NumpyRate)
        print("Precision: ",np.linalg.norm(e_np-C_np),
              np.linalg.norm(f_np-D_np)) 
        
    # Native & Numba Implementation
    if NumbaMethod:
        print("Performing Numba implementation")
        TimeIn=time.time()
        g_np,h_np=NumbaDFT(a_np,b_np)
        NumbaElapsed=time.time()-TimeIn
        NumbaRate=int(SIZE/NumbaElapsed)
        print("NumbaRate: %i" % NumbaRate)
        print("Precision: ",np.linalg.norm(g_np-C_np),
              np.linalg.norm(h_np-D_np)) 
    
    # OpenCL Implementation
    if OpenCLMethod:
        print("Performing OpenCL implementation")
        TimeIn=time.time()
        i_np,j_np=OpenCLDFT(a_np,b_np,Device)
        OpenCLElapsed=time.time()-TimeIn
        OpenCLRate=int(SIZE/OpenCLElapsed)
        print("OpenCLRate: %i" % OpenCLRate)
        print("Precision: ",np.linalg.norm(i_np-C_np),
              np.linalg.norm(j_np-D_np)) 
    
    # CUDA Implementation
    if CUDAMethod:
        print("Performing CUDA implementation")
        TimeIn=time.time()
        k_np,l_np=CUDADFT(a_np,b_np,Device,Threads)
        CUDAElapsed=time.time()-TimeIn
        CUDARate=int(SIZE/CUDAElapsed)
        print("CUDARate: %i" % CUDARate)
        print("Precision: ",np.linalg.norm(k_np-C_np),
              np.linalg.norm(l_np-D_np)) 
    
    if NumpyFFTMethod:
        print("Performing NumpyFFT implementation")
        TimeIn=time.time()
        m_np,n_np=NumpyFFT(a_np,b_np)
        NumpyFFTElapsed=time.time()-TimeIn
        NumpyFFTRate=int(SIZE/NumpyFFTElapsed)
        print("NumpyFFTElapsed: %i" % NumpyFFTElapsed)
        print("NumpyFFTRate: %i" % NumpyFFTRate)
        print("Precision: ",np.linalg.norm(m_np-C_np),
              np.linalg.norm(n_np-D_np)) 
    
    # GPyFFT Implementation
    if GPyFFTMethod:
        print("Performing OpenCLFFT implementation")
        TimeIn=time.time()
        i_np,j_np=GPyFFT(a_np,b_np,Device)
        GPyFFTElapsed=time.time()-TimeIn
        GPyFFTRate=int(SIZE/GPyFFTElapsed)
        print("OpenCLFFTElapsed: %i" % GPyFFTElapsed)
        print("GPyFFTRate: %i" % GPyFFTRate)
        print("Precision: ",np.linalg.norm(i_np-C_np),
              np.linalg.norm(j_np-D_np)) 
    
    # OpenCL Implementation
    if OpenCLFFTMethod:
        print("Performing OpenCLFFT implementation")
        TimeIn=time.time()
        i_np,j_np=OpenCLFFT(a_np,b_np,Device)
        OpenCLFFTElapsed=time.time()-TimeIn
        OpenCLFFTRate=int(SIZE/OpenCLFFTElapsed)
        print("OpenCLFFTElapsed: %i" % OpenCLFFTElapsed)
        print("OpenCLFFTRate: %i" % OpenCLFFTRate)
        print("Precision: ",np.linalg.norm(i_np-C_np),
              np.linalg.norm(j_np-D_np)) 
    
    if OpenCLMethod and NumpyFFTMethod:
        print(OpenCLMethod,NumpyFFTMethod)
        print("Precision: ",np.linalg.norm(m_np-i_np),
              np.linalg.norm(n_np-j_np)) 
        print((m_np-i_np),(n_np-j_np)) 
        print(i_np,j_np) 
        print(m_np,n_np) 
        print((i_np-m_np),(j_np-n_np)) 
        
    if CUDAMethod and NumpyFFTMethod:
        print(CUDAMethod,NumpyFFTMethod)
        print("Precision: ",np.linalg.norm(m_np-k_np),
              np.linalg.norm(n_np-l_np)) 
        print((m_np-k_np),(n_np-l_np)) 
        print(k_np,l_np) 
        print(m_np,n_np) 
        print((k_np-m_np),(l_np-n_np)) 
        
    if OpenCLMethod and NumpyMethod:
        print(OpenCLMethod,NumpyMethod)
        print("Precision: ",np.linalg.norm(e_np-i_np),
              np.linalg.norm(f_np-j_np)) 
        print((e_np-i_np),(f_np-j_np)) 
        
    if NumpyFFTMethod and NumpyMethod:
        print(NumpyFFTMethod,NumpyMethod)
        print("Precision: ",np.linalg.norm(e_np-m_np),
              np.linalg.norm(f_np-n_np)) 
        print(e_np,f_np) 
        print(m_np,n_np) 
        print((e_np-m_np),(f_np-n_np)) 
        
    if NumpyFFTMethod and NaiveMethod:
        print(NumpyFFTMethod,NaiveMethod)
        print("Precision: ",np.linalg.norm(c_np-m_np),
              np.linalg.norm(d_np-n_np)) 
        print(c_np,d_np) 
        print(m_np,n_np) 
        print((c_np-m_np),(d_np-n_np)) 
        
    if NumpyFFTMethod and NumbaMethod:
        print(NumpyFFTMethod,NumbaMethod)
        print("Precision: ",np.linalg.norm(g_np-m_np),
              np.linalg.norm(h_np-n_np)) 
        print(g_np,h_np) 
        print(m_np,n_np) 
        print((g_np-m_np),(h_np-n_np)) 
        
    if OpenCLFFTMethod and NumpyFFTMethod:
        print("NumpyOpenCLRatio: %f" % (OpenCLFFTRate/NumpyFFTRate))

    if GPyFFTMethod and NumpyFFTMethod:
        print("NumpyGPyFFTRatio: %f" % (GPyFFTRate/NumpyFFTRate))
