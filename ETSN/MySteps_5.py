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
def NativeSillyAddition(a_np,b_np):
    return(MySillyFunction(a_np)+MySillyFunction(b_np))

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
def CUDASillyAddition(a_np,b_np):
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

__global__ void sillysum(float *dest, float *a, float *b)
{
  const int i = blockIdx.x;
  dest[i] = MySillyFunction(a[i]) + MySillyFunction(b[i]);
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

    TimeIn=time.time()
    sillysum(drv.Out(res_np), drv.In(a_np), drv.In(b_np),
             block=(1,1,1), grid=(a_np.size,1))
    Elapsed=time.time()-TimeIn
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
def OpenCLSillyAddition(a_np,b_np):

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

float MySillyFunction(float x)
{
    return(pow(sqrt(log(exp(atanh(tanh(asinh(sinh(acosh(cosh(atan(tan(asin(sin(acos(cos(x))))))))))))))),2)); 
}

__kernel void sillysum(
    __global const float *a_g, __global const float *b_g, __global float *res_g)
{
  int gid = get_global_id(0);
  res_g[gid] = MySillyFunction(a_g[gid]) + MySillyFunction(b_g[gid]);
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
    CallCL=knl(queue, a_np.shape, None, a_g, b_g, res_g)
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

    # Size of input vectors definition based on stdin
    import sys
    try:
        SIZE=int(sys.argv[1])
        print("Size of vectors set to %i" % SIZE)
    except: 
        SIZE=50000
        print("Size of vectors set to default size %i" % SIZE)
        
    a_np = np.random.rand(SIZE).astype(np.float32)
    b_np = np.random.rand(SIZE).astype(np.float32)

    # Native Implementation
    TimeIn=time.time()
    # res_np=NativeAddition(a_np,b_np)
    res_np=NativeSillyAddition(a_np,b_np)
    NativeElapsed=time.time()-TimeIn
    NativeRate=int(SIZE/NativeElapsed)
    print("NativeRate: %i" % NativeRate)

    # OpenCL Implementation
    TimeIn=time.time()
    # res_cl=OpenCLAddition(a_np,b_np)
    res_cl=OpenCLSillyAddition(a_np,b_np)
    OpenCLElapsed=time.time()-TimeIn
    OpenCLRate=int(SIZE/OpenCLElapsed)
    print("OpenCLRate: %i" % OpenCLRate)

    # CUDA Implementation
    TimeIn=time.time()
    # res_cuda=CUDAAddition(a_np,b_np)
    res_cuda=CUDASillyAddition(a_np,b_np)
    CUDAElapsed=time.time()-TimeIn
    CUDARate=int(SIZE/CUDAElapsed)
    print("CUDARate: %i" % CUDARate)
    
    print("OpenCLvsNative ratio: %f" % (OpenCLRate/NativeRate))
    print("CUDAvsNative ratio: %f" % (CUDARate/NativeRate))
    
   # Check on OpenCL with Numpy:
    print(res_cl - res_np)
    print(np.linalg.norm(res_cl - res_np))
    try:
        assert np.allclose(res_np, res_cl)
    except:
        print("Results between Native & OpenCL seem to be too different!")
        
    # Check on CUDA with Numpy:
    print(res_cuda - res_np)
    print(np.linalg.norm(res_cuda - res_np))
    try:
        assert np.allclose(res_np, res_cuda)
    except:
        print("Results between Native & CUDA seem to be too different!")


