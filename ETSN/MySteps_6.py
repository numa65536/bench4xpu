#!/usr/bin/env python3

import numpy as np
import pyopencl as cl

# piling 16 arithmetical functions
def MySillyFunction(x):
    return(np.power(np.sqrt(np.log(np.exp(np.arctanh(np.tanh(np.arcsinh(np.sinh(np.arccosh(np.cosh(np.arctan(np.tan(np.arcsin(np.sin(np.arccos(np.cos(x))))))))))))))),2))

# Native Operation under Numpy (for prototyping & tests
def NativeAddition(a_np,b_np):
    return(a_np+b_np)

# Native Operation with MySillyFunction under Numpy (for prototyping & tests
def NativeSillyAddition(a_np,b_np,Calls):
    a=a_np.copy()
    b=b_np.copy()
    for i in range(Calls):
        a=MySillyFunction(a)
        b=MySillyFunction(b)
        
    return(a+b)

# CUDA complete operation
def CUDAAddition(a_np,b_np):
    import pycuda.autoinit
    import pycuda.driver as drv
    import numpy

    from pycuda.compiler import SourceModule
    mod = SourceModule("""
    __global__ void sum(float *dest, float *a, float *b)
{
  // const int i = threadIdx.x;
  const int i = blockIdx.x;
  dest[i] = a[i] + b[i];
}
""")

    # sum = mod.get_function("sum")
    sum = mod.get_function("sum")

    res_np = numpy.zeros_like(a_np)
    sum(drv.Out(res_np), drv.In(a_np), drv.In(b_np),
        block=(1,1,1), grid=(a_np.size,1))
    return(res_np)

# CUDA Silly complete operation
def CUDASillyAddition(a_np,b_np,Calls,Threads):
    import pycuda.autoinit
    import pycuda.driver as drv
    import numpy

    from pycuda.compiler import SourceModule
    TimeIn=time.time()
    mod = SourceModule("""
__device__ float MySillyFunction(float x)
{
    return(pow(sqrt(log(exp(atanh(tanh(asinh(sinh(acosh(cosh(atan(tan(asin(sin(acos(cos(x))))))))))))))),2)); 
}

__global__ void sillysum(float *dest, float *a, float *b, int Calls)
{
  const int i = blockDim.x*blockIdx.x+threadIdx.x;
  float ai=a[i];
  float bi=b[i];

  for (int c=0;c<Calls;c++)
  {
    ai=MySillyFunction(ai);
    bi=MySillyFunction(bi);
  }

  dest[i] = ai + bi;
}
""")
    Elapsed=time.time()-TimeIn
    print("Definition of kernel : %.3f" % Elapsed)

    TimeIn=time.time()
    # sum = mod.get_function("sum")
    sillysum = mod.get_function("sillysum")
    Elapsed=time.time()-TimeIn
    print("Synthesis of kernel : %.3f" % Elapsed)

    TimeIn=time.time()
    res_np = numpy.zeros_like(a_np)
    Elapsed=time.time()-TimeIn
    print("Allocation on Host for results : %.3f" % Elapsed)

    Size=a_np.size
    if (Size % Threads != 0):
        print("Impossible : %i not multiple of %i..." % (Threads,Size) )
        TimeIn=time.time()
        sillysum(drv.Out(res_np), drv.In(a_np), drv.In(b_np), np.uint32(Calls),
                 block=(1,1,1), grid=(a_np.size,1))
        Elapsed=time.time()-TimeIn
        print("Execution of kernel : %.3f" % Elapsed)
    else:
        Blocks=int(Size/Threads)
        TimeIn=time.time()
        sillysum(drv.Out(res_np), drv.In(a_np), drv.In(b_np), np.uint32(Calls),
                 block=(Threads,1,1), grid=(Blocks,1))
        Elapsed=time.time()-TimeIn
        print("Execution of kernel : %.3f" % Elapsed)
        
    print("Execution of kernel : %.3f" % Elapsed)
    return(res_np)

# OpenCL complete operation
def OpenCLAddition(a_np,b_np):

    # Context creation
    ctx = cl.create_some_context()
    # Every process is stored in a queue
    queue = cl.CommandQueue(ctx)

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
__kernel void sum(
    __global const float *a_g, __global const float *b_g, __global float *res_g)
{
  int gid = get_global_id(0);
  res_g[gid] = a_g[gid] + b_g[gid];
}
""").build()
    Elapsed=time.time()-TimeIn
    print("Building kernels : %.3f" % Elapsed)
    
    TimeIn=time.time()
    # Memory allocation on Device for result
    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
    Elapsed=time.time()-TimeIn
    print("Allocation on Device for results : %.3f" % Elapsed)

    TimeIn=time.time()
    # Synthesis of function "sum" inside Kernel Sources
    knl = prg.sum  # Use this Kernel object for repeated calls
    Elapsed=time.time()-TimeIn
    print("Synthesis of kernel : %.3f" % Elapsed)

    TimeIn=time.time()
    # Call of kernel previously defined 
    knl(queue, a_np.shape, None, a_g, b_g, res_g)
    Elapsed=time.time()-TimeIn
    print("Execution of kernel : %.3f" % Elapsed)

    TimeIn=time.time()
    # Creation of vector for result with same size as input vectors
    res_np = np.empty_like(a_np)
    Elapsed=time.time()-TimeIn
    print("Allocation on Host for results: %.3f" % Elapsed)

    TimeIn=time.time()
    # Copy from Device to Host
    cl.enqueue_copy(queue, res_np, res_g)
    Elapsed=time.time()-TimeIn
    print("Copy from Device 2 Host : %.3f" % Elapsed)

    # Liberation of memory
    a_g.release()
    b_g.release()
    res_g.release()
    
    return(res_np)

# OpenCL complete operation
def OpenCLSillyAddition(a_np,b_np,Calls):

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

float MySillyFunction(float x)
{
    return(pow(sqrt(log(exp(atanh(tanh(asinh(sinh(acosh(cosh(atan(tan(asin(sin(acos(cos(x))))))))))))))),2)); 
}

__kernel void sillysum(
    __global const float *a_g, __global const float *b_g, __global float *res_g, int Calls)
{
  int gid = get_global_id(0);
  float ai=a_g[gid];
  float bi=b_g[gid];

  for (int c=0;c<Calls;c++)
  {
    ai=MySillyFunction(ai);
    bi=MySillyFunction(bi);
  }

  res_g[gid] = ai + bi;
}

__kernel void sum(
    __global const float *a_g, __global const float *b_g, __global float *res_g)
{
  int gid = get_global_id(0);
  res_g[gid] = a_g[gid] + b_g[gid];
}
""").build()
    Elapsed=time.time()-TimeIn
    print("Building kernels : %.3f" % Elapsed)
    
    TimeIn=time.time()
    # Memory allocation on Device for result
    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
    Elapsed=time.time()-TimeIn
    print("Allocation on Device for results : %.3f" % Elapsed)

    TimeIn=time.time()
    # Synthesis of function "sillysum" inside Kernel Sources
    knl = prg.sillysum  # Use this Kernel object for repeated calls
    Elapsed=time.time()-TimeIn
    print("Synthesis of kernel : %.3f" % Elapsed)

    TimeIn=time.time()
    # Call of kernel previously defined 
    CallCL=knl(queue, a_np.shape, None, a_g, b_g, res_g, np.uint32(Calls))
    # 
    CallCL.wait()
    Elapsed=time.time()-TimeIn
    print("Execution of kernel : %.3f" % Elapsed)

    TimeIn=time.time()
    # Creation of vector for result with same size as input vectors
    res_np = np.empty_like(a_np)
    Elapsed=time.time()-TimeIn
    print("Allocation on Host for results: %.3f" % Elapsed)

    TimeIn=time.time()
    # Copy from Device to Host
    cl.enqueue_copy(queue, res_np, res_g)
    Elapsed=time.time()-TimeIn
    print("Copy from Device 2 Host : %.3f" % Elapsed)

    # Liberation of memory
    a_g.release()
    b_g.release()
    res_g.release()
    
    return(res_np)

import sys
import time

if __name__=='__main__':

    GpuStyle='OpenCL'
    SIZE=1024
    Device=0
    Calls=1
    Threads=1
    Serial=True
    
    import getopt

    HowToUse='%s -n -g <CUDA/OpenCL> -s <SizeOfVector> -d <DeviceId> -c <SillyCalls> -t <Threads>'
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hng:s:d:c:t:",["gpustyle=","size=","device=","calls=","threads="])
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
        elif opt in ("-t", "--threads"):
            Threads=int(arg)
        elif opt in ("-c", "--calls"):
            Calls=int(arg)
        elif opt in ("-g", "--gpustyle"):
            GpuStyle = arg
        elif opt in ("-s", "--size"):
            SIZE = int(arg)
        elif opt in ("-n"):
            Serial = False

    print("Device Selection : %i" % Device)
    print("GpuStyle used : %s" % GpuStyle)
    print("Size of complex vector : %i" % SIZE)
    print("Number of silly calls : %i" % Calls)
    print("Number of Threads : %i" % Threads)
    print("Serial compute : %i" % Serial)

    if GpuStyle=='CUDA':
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

    if GpuStyle=='OpenCL':
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
        
    a_np = np.random.rand(SIZE).astype(np.float32)
    b_np = np.random.rand(SIZE).astype(np.float32)

    # Native Implementation
    if Serial:
        TimeIn=time.time()
        res_np=NativeSillyAddition(a_np,b_np,Calls)
        NativeElapsed=time.time()-TimeIn
        NativeRate=int(SIZE/NativeElapsed)
        print("NativeRate: %i" % NativeRate)

    # OpenCL Implementation
    if GpuStyle=='OpenCL' or GpuStyle=='all':
        TimeIn=time.time()
        # res_cl=OpenCLAddition(a_np,b_np)
        res_cl=OpenCLSillyAddition(a_np,b_np,Calls)
        OpenCLElapsed=time.time()-TimeIn
        OpenCLRate=int(SIZE/OpenCLElapsed)
        print("OpenCLRate: %i" % OpenCLRate)
        # Check on OpenCL with Numpy:
        if Serial:
            print(res_cl - res_np)
            print(np.linalg.norm(res_cl - res_np))
            try:
                assert np.allclose(res_np, res_cl)
            except:
                print("Results between Native & OpenCL seem to be too different!")

            print("OpenCLvsNative ratio: %f" % (OpenCLRate/NativeRate))
            
    # CUDA Implementation
    if GpuStyle=='CUDA' or GpuStyle=='all':
        TimeIn=time.time()
        res_cuda=CUDASillyAddition(a_np,b_np,Calls,Threads)
        CUDAElapsed=time.time()-TimeIn
        CUDARate=int(SIZE/CUDAElapsed)
        print("CUDARate: %i" % CUDARate)
        # Check on CUDA with Numpy:
        if Serial:
            print(res_cuda - res_np)
            print(np.linalg.norm(res_cuda - res_np))
            try:
                assert np.allclose(res_np, res_cuda)
            except:
                print("Results between Native & CUDA seem to be too different!")
    
            print("CUDAvsNative ratio: %f" % (CUDARate/NativeRate))
    
        


