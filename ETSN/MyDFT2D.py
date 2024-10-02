#!/usr/bin/env python3

import numpy as np
import pyopencl as cl
from numpy import pi,cos,sin

# Naive Discrete Fourier Transform
def MyDFT(x,y):
    size=x.shape[0]
    X=np.zeros(x.shape).astype(np.float32)
    Y=np.zeros(x.shape).astype(np.float32)
    for k in range(size):
        for l in range(size):
            for i in range(size):
                for j in range(size):
                    t=np.float32(2*pi*((i*k)/size+(l*j)/size))
                    X[k,l]+=x[i,j]*cos(t)+y[i,j]*sin(t)
                    Y[k,l]+=-x[i,j]*sin(t)+y[i,j]*cos(t)
    return(X,Y)

#
def NumpyFFT(x,y):
    xy=np.csingle(x+1.j*y)
    XY=np.fft.fft2(xy)
    return(XY.real,XY.imag)

def OpenCLFFT(x,y,device):
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
    transform = FFT(ctx, queue, XY_gpu, axes=(0,1))
    event, = transform.enqueue()
    event.wait()
    Elapsed=time.time()-TimeIn
    print("Compute FFT : %.3f" % Elapsed)
    TimeIn=time.time()
    XY = XY_gpu.get()
    Elapsed=time.time()-TimeIn
    print("Copy from Device to Host : %.3f" % Elapsed)

    return(XY.real,XY.imag)

# # Numpy Discrete Fourier Transform
# def NumpyDFT(x,y):
#     size=x.shape[0]
#     X=np.zeros([size,size]).astype(np.float32)
#     Y=np.zeros([size,size]).astype(np.float32)
#     nj=np.multiply(2.0*np.pi/size,np.arange(size)).astype(np.float32)
#     for k in range(size):
#         for l in range(size):
#         X[k]=np.sum(np.subtract(np.multiply(np.cos(k*nj),x),np.multiply(np.sin(k*nj),y)))
#         Y[k]=np.sum(np.add(np.multiply(np.sin(k*nj),x),np.multiply(np.cos(k*nj),y)))
#     return(X,Y)

# Numba Discrete Fourier Transform
import numba
@numba.njit(parallel=True)
def NumbaDFT(x,y):
    size=x.shape[0]
    X=np.zeros(x.shape).astype(np.float32)
    Y=np.zeros(y.shape).astype(np.float32)
    for k in numba.prange(size):
        for l in numba.prange(size):
            for i in numba.prange(size):
                for j in numba.prange(size):
                    t=np.float32(2*pi*((i*k)/size+(l*j)/size))
                    X[k,l]+=x[i,j]*cos(t)+y[i,j]*sin(t)
                    Y[k,l]+=-x[i,j]*sin(t)+y[i,j]*cos(t)
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
  int gidx = get_global_id(0);
  int gidy = get_global_id(1);
  uint size = get_global_size(0);
  float A=0.,B=0.;
  for (uint i=0; i<size;i++) for (uint j=0; j<size;j++) 
  {
     float angle=2.*PI*((float)(gidx*i)/(float)size+
                        (float)(gidy*j)/(float)size);
     A+=a_g[i+size*j]*cos(angle)+b_g[i+size*j]*sin(angle);
     B+=-a_g[i+size*j]*sin(angle)+b_g[i+size*j]*cos(angle);
  }
  A_g[gidx+size*gidy]=A;
  B_g[gidx+size*gidy]=B;
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
  const int gidx = blockIdx.x*blockDim.x+threadIdx.x;
  const int gidy = blockIdx.y*blockDim.y+threadIdx.y;
  uint sizex = gridDim.x*blockDim.x;
  uint sizey = gridDim.y*blockDim.y;
  uint size = gridDim.x*blockDim.x*gridDim.y*blockDim.y;
  float A=0.,B=0.;
  for (uint i=0; i<sizex;i++) for (uint j=0; j<sizey;j++) 
  {
     float angle=2.*PI*((float)(gidx*i)/(float)sizex+
                        (float)(gidy*j)/(float)sizey);
     A+=a_g[i+sizex*j]*cos(angle)+b_g[i+sizex*j]*sin(angle);
     B+=-a_g[i+sizex*j]*sin(angle)+b_g[i+sizex*j]*cos(angle);
  }
  A_g[gidx+sizey*gidy]=A;
  B_g[gidx+sizey*gidy]=B;
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

    Size=a_np.shape
    if (Size[0] % Threads != 0):
        print("Impossible : %i not multiple of %i..." % (Threads,Size[0]) )
        TimeIn=time.time()
        MyDFT(drv.Out(A_np), drv.Out(B_np), drv.In(a_np), drv.In(b_np),
              block=(1,1,1), grid=Size)
        Elapsed=time.time()-TimeIn
        print("Execution of kernel : %.3f" % Elapsed)
    else:
        Blocks=(int(Size[0]/Threads),int(Size[1]/Threads));
        TimeIn=time.time()
        MyDFT(drv.Out(A_np), drv.Out(B_np), drv.In(a_np), drv.In(b_np),
              block=(Threads,Threads,1), grid=Blocks)
        Elapsed=time.time()-TimeIn
        print("Execution of kernel : %.3f" % Elapsed)
        
    Context.pop()
    Context.detach()
    
    return(A_np,B_np)

import sys
import time

if __name__=='__main__':

    SIZE=4
    Device=0
    NaiveMethod=False
    NumpyMethod=False
    NumpyFFTMethod=True
    NumbaMethod=False
    OpenCLMethod=False
    OpenCLFFTMethod=True
    CUDAMethod=False
    Threads=1
    Verbose=False
    
    import getopt

    HowToUse='%s -v [Verbose] -n [Naive] -y [numpYFFT] -a [numbA] -o [OpenCL] -g [OpenCLFFT] -c [CUDA] -s <SizeOfVector> -d <DeviceId> -t <threads>'
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"gvnyaochs:d:t:",["size=","device="])
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
            NumpyFFTMethod=True
        elif opt in ("-a"):
            NumbaMethod=True
        elif opt in ("-o"):
            OpenCLMethod=True
        elif opt in ("-g"):
            OpenCLFFTMethod=True
        elif opt in ("-c"):
            CUDAMethod=True
        elif opt in ("-v"):
            Verbose=True

    print("Device Selection : %i" % Device)
    print("Size of complex vector : %i" % SIZE)
    print("Verbosity %s " % Verbose )
    print("DFT Naive computation %s " % NaiveMethod )
    print("DFT Numpy computation %s " % NumpyMethod )
    print("FFT Numpy computation %s " % NumpyFFTMethod )
    print("DFT Numba computation %s " % NumbaMethod )
    print("DFT OpenCL computation %s " % OpenCLMethod )
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

    
        
    a_np = np.ones([SIZE,SIZE]).astype(np.float32)
    b_np = np.ones([SIZE,SIZE]).astype(np.float32)
    # a_np = np.zeros([SIZE,SIZE]).astype(np.float32)
    # b_np = np.zeros([SIZE,SIZE]).astype(np.float32)
    # a_np[0,0]=1;

    np.set_printoptions(precision=1,suppress=True)

    # print(a_np+1.j*b_np)
    
    # print(np.fft.fft2(a_np+1.j*b_np))
    
    C_np = np.zeros([SIZE,SIZE]).astype(np.float32)
    D_np = np.zeros([SIZE,SIZE]).astype(np.float32)
    C_np[0,0] = np.float32(SIZE*SIZE)
    D_np[0,0] = np.float32(SIZE*SIZE)
    
    # Native & Naive Implementation
    if NaiveMethod:
        print("Performing naive implementation")
        TimeIn=time.time()
        c_np,d_np=MyDFT(a_np,b_np)
        NativeElapsed=time.time()-TimeIn
        NativeRate=int(SIZE*SIZE/NativeElapsed)
        print("NativeElapsed: %i" % NativeElapsed)
        print("NativeRate: %i" % NativeRate)
        print("Precision: ",np.linalg.norm(c_np-C_np),
              np.linalg.norm(d_np-D_np))
        if Verbose:
            print(c_np+1.j*d_np)
        
    # Native & Numpy Implementation
    if NumpyFFTMethod:
        print("Performing Numpy FFT implementation")
        TimeIn=time.time()
        e_np,f_np=NumpyFFT(a_np,b_np)
        NumpyFFTElapsed=time.time()-TimeIn
        NumpyFFTRate=int(SIZE*SIZE/NumpyFFTElapsed)
        print("NumpyFFTElapsed: %i" % NumpyFFTElapsed)
        print("NumpyFFTRate: %i" % NumpyFFTRate)
        print("Precision: ",np.linalg.norm(e_np-C_np),
              np.linalg.norm(f_np-D_np)) 
        if Verbose:
            print(e_np+1.j*f_np)
        
    # Native & Numba Implementation
    if NumbaMethod:
        print("Performing Numba implementation")
        TimeIn=time.time()
        g_np,h_np=NumbaDFT(a_np,b_np)
        NumbaElapsed=time.time()-TimeIn
        NumbaRate=int(SIZE*SIZE/NumbaElapsed)
        print("NumbaElapsed: %i" % NumbaElapsed)
        print("NumbaRate: %i" % NumbaRate)
        print("Precision: ",np.linalg.norm(g_np-C_np),
              np.linalg.norm(h_np-D_np)) 
        if Verbose:
            print(g_np+1.j*h_np)
    
    # OpenCL Implementation
    if OpenCLMethod:
        print("Performing OpenCL implementation")
        TimeIn=time.time()
        i_np,j_np=OpenCLDFT(a_np,b_np,Device)
        OpenCLElapsed=time.time()-TimeIn
        OpenCLRate=int(SIZE*SIZE/OpenCLElapsed)
        print("OpenCLElapsed: %i" % OpenCLElapsed)
        print("OpenCLRate: %i" % OpenCLRate)
        print("Precision: ",np.linalg.norm(i_np-C_np),
              np.linalg.norm(j_np-D_np))
        if Verbose:
            print(i_np+1.j*j_np)
        
    # CUDA Implementation
    if CUDAMethod:
        print("Performing CUDA implementation")
        TimeIn=time.time()
        k_np,l_np=CUDADFT(a_np,b_np,Device,Threads)
        CUDAElapsed=time.time()-TimeIn
        CUDARate=int(SIZE*SIZE/CUDAElapsed)
        print("CUDAElapsed: %i" % CUDAElapsed)
        print("CUDARate: %i" % CUDARate)
        print("Precision: ",np.linalg.norm(k_np-C_np),
              np.linalg.norm(l_np-D_np)) 
        if Verbose:
            print(k_np+1.j*l_np)

    # OpenCL Implementation
    if OpenCLFFTMethod:
        print("Performing OpenCL FFT implementation")
        TimeIn=time.time()
        m_np,n_np=OpenCLFFT(a_np,b_np,Device)
        OpenCLFFTElapsed=time.time()-TimeIn
        OpenCLFFTRate=int(SIZE*SIZE/OpenCLFFTElapsed)
        print("OpenCLFFTElapsed: %i" % OpenCLFFTElapsed)
        print("OpenCLFFTRate: %i" % OpenCLFFTRate)
        print("Precision: ",np.linalg.norm(m_np-C_np),
              np.linalg.norm(n_np-D_np))
        if Verbose:
            print(m_np+1.j*n_np)
        
