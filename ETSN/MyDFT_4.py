#!/usr/bin/env python3

import numpy as np
import pyopencl as cl
from numpy import pi,cos,sin

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
        Y[i]=np.sum(np.subtract(np.multiply(np.cos(i*nj),y),np.multiply(np.sin(i*nj),x)))
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
        Y[i]=np.sum(np.subtract(np.multiply(np.cos(i*nj),y),np.multiply(np.sin(i*nj),x)))
    return(X,Y)

# OpenCL complete operation
def OpenCLDFT(a_np,b_np):

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

import sys
import time

if __name__=='__main__':

    # Size of input vectors definition based on stdin
    import sys
    try:
        SIZE=int(sys.argv[1])
        print("Size of vectors set to %i" % SIZE)
    except: 
        SIZE=256
        print("Size of vectors set to default size %i" % SIZE)
        
    a_np = np.ones(SIZE).astype(np.float32)
    b_np = np.ones(SIZE).astype(np.float32)

    C_np = np.zeros(SIZE).astype(np.float32)
    D_np = np.zeros(SIZE).astype(np.float32)
    C_np[0] = np.float32(SIZE)
    D_np[0] = np.float32(SIZE)
    
    # # Native & Naive Implementation
    # print("Performing naive implementation")
    # TimeIn=time.time()
    # c_np,d_np=MyDFT(a_np,b_np)
    # NativeElapsed=time.time()-TimeIn
    # NativeRate=int(SIZE/NativeElapsed)
    # print("NativeRate: %i" % NativeRate)
    # print("Precision: ",np.linalg.norm(c_np-C_np),np.linalg.norm(d_np-D_np)) 

    # Native & Numpy Implementation
    print("Performing Numpy implementation")
    TimeIn=time.time()
    e_np,f_np=NumpyDFT(a_np,b_np)
    NumpyElapsed=time.time()-TimeIn
    NumpyRate=int(SIZE/NumpyElapsed)
    print("NumpyRate: %i" % NumpyRate)
    print("Precision: ",np.linalg.norm(e_np-C_np),np.linalg.norm(f_np-D_np)) 
        
    # Native & Numba Implementation
    print("Performing Numba implementation")
    TimeIn=time.time()
    g_np,h_np=NumbaDFT(a_np,b_np)
    NumbaElapsed=time.time()-TimeIn
    NumbaRate=int(SIZE/NumbaElapsed)
    print("NumbaRate: %i" % NumbaRate)
    print("Precision: ",np.linalg.norm(g_np-C_np),np.linalg.norm(h_np-D_np)) 
    
    # OpenCL Implementation
    print("Performing OpenCL implementation")
    TimeIn=time.time()
    i_np,j_np=OpenCLDFT(a_np,b_np)
    OpenCLElapsed=time.time()-TimeIn
    OpenCLRate=int(SIZE/OpenCLElapsed)
    print("OpenCLRate: %i" % OpenCLRate)
    print("Precision: ",np.linalg.norm(i_np-C_np),np.linalg.norm(j_np-D_np)) 
    
