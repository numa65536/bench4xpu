#!/usr/bin/env python3

import numpy as np
import pyopencl as cl

# Native Operation under Numpy (for prototyping & tests
def NativeAddition(a_np,b_np):
    return(a_np+b_np)

# OpenCL complete operation
def OpenCLAddition(a_np,b_np):

    # Context creation
    ctx = cl.create_some_context()
    # Every process is stored in a queue
    queue = cl.CommandQueue(ctx)

    # Copy from Host to Device using pointers
    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

    # Definition of kernel under OpenCL
    prg = cl.Program(ctx, """
__kernel void sum(
    __global const float *a_g, __global const float *b_g, __global float *res_g)
{
  int gid = get_global_id(0);
  res_g[gid] = a_g[gid] + b_g[gid];
}
""").build()

    # Memory allocation on Device for result
    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
    # Synthesis of function "sum" inside Kernel Sources
    knl = prg.sum  # Use this Kernel object for repeated calls
    # Call of kernel previously defined 
    knl(queue, a_np.shape, None, a_g, b_g, res_g)

    # Creation of vector for result with same size as input vectors
    res_np = np.empty_like(a_np)
    # Copy from Device to Host
    cl.enqueue_copy(queue, res_np, res_g)
    
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

    TimeIn=time.time()
    res_np=NativeAddition(a_np,b_np)
    NativeElapsed=time.time()-TimeIn
    NativeRate=int(SIZE/NativeElapsed)
    print("NativeRate: %i" % NativeRate)

    TimeIn=time.time()
    res_cl=OpenCLAddition(a_np,b_np)
    OpenCLElapsed=time.time()-TimeIn
    OpenCLRate=int(SIZE/OpenCLElapsed)
    print("OpenCLRate: %i" % OpenCLRate)
    
    print("OpenCLvsNative ratio: %f" % (OpenCLRate/NativeRate))
    
    # Check on CPU with Numpy:
    print(res_cl - res_np)
    print(np.linalg.norm(res_cl - res_np))
    assert np.allclose(res_np, res_cl)
