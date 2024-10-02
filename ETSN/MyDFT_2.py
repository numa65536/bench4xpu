#!/usr/bin/env python3

import numpy as np
import pyopencl as cl

# Naive Discrete Fourier Transform
def MyDFT(x,y):
    from numpy import pi,cos,sin
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
    from numpy import pi,cos,sin
    size=x.shape[0]
    X=np.zeros(size).astype(np.float32)
    Y=np.zeros(size).astype(np.float32)
    nj=np.multiply(2.0*np.pi/size,np.arange(size)).astype(np.float32)
    for i in range(size):
        X[i]=np.sum(np.add(np.multiply(np.cos(i*nj),x),np.multiply(np.sin(i*nj),y)))
        Y[i]=np.sum(np.subtract(np.multiply(np.cos(i*nj),y),np.multiply(np.sin(i*nj),x)))
    return(X,Y)

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
    
    # Native & Naive Implementation
    print("Performing naive implementation")
    TimeIn=time.time()
    c_np,d_np=MyDFT(a_np,b_np)
    NativeElapsed=time.time()-TimeIn
    NativeRate=int(SIZE/NativeElapsed)
    print("NativeRate: %i" % NativeRate)
    print("Precision: ",np.linalg.norm(c_np-C_np),np.linalg.norm(d_np-D_np)) 
    
    # Native & Numpy Implementation
    print("Performing Numpy implementation")
    TimeIn=time.time()
    e_np,f_np=NumpyDFT(a_np,b_np)
    NumpyElapsed=time.time()-TimeIn
    NumpyRate=int(SIZE/NumpyElapsed)
    print("NumpyRate: %i" % NumpyRate)
    print("Precision: ",np.linalg.norm(e_np-C_np),np.linalg.norm(f_np-D_np)) 
