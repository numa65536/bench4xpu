#!/usr/bin/env python3

#
# Pi-by-MonteCarlo using PyCUDA/PyOpenCL
#
# CC BY-NC-SA 2011 : Emmanuel QUEMENER <emmanuel.quemener@gmail.com> 
# Cecill v2 : Emmanuel QUEMENER <emmanuel.quemener@gmail.com>
#
# Thanks to Andreas Klockner for PyCUDA:
# http://mathema.tician.de/software/pycuda
# Thanks to Andreas Klockner for PyOpenCL:
# http://mathema.tician.de/software/pyopencl
# 

# 2013-01-01 : problems with launch timeout
# http://stackoverflow.com/questions/497685/how-do-you-get-around-the-maximum-cuda-run-time
# Option "Interactive" "0" in /etc/X11/xorg.conf

# Common tools
import numpy
from numpy.random import randint as nprnd
import sys
import getopt
import time
import math
import itertools
from socket import gethostname

from threading import Thread

from PiXPU import *

class threadWithReturn(Thread):
    def __init__(self, *args, **kwargs):
        super(threadWithReturn, self).__init__(*args, **kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args, **kwargs):
        super(threadWithReturn, self).join(*args, **kwargs)
        return self._return

if __name__=='__main__':
    
    # Set defaults values
  
    # Id of Device : 1 is for first find !
    Device=1
    # GPU style can be Cuda (Nvidia implementation) or OpenCL
    GpuStyle='OpenCL'
    # Iterations is integer
    Iterations=10000000
    # BlocksBlocks in first number of Blocks to explore
    BlocksBegin=1
    # BlocksEnd is last number of Blocks to explore
    BlocksEnd=16
    # BlocksStep is the step of Blocks to explore
    BlocksStep=1
    # ThreadsBlocks in first number of Blocks to explore
    ThreadsBegin=1
    # ThreadsEnd is last number of Blocks to explore
    ThreadsEnd=1
    # ThreadsStep is the step of Blocks to explore
    ThreadsStep=1
    # Redo is the times to redo the test to improve metrology
    Redo=1
    # OutMetrology is method for duration estimation : False is GPU inside
    OutMetrology=False
    Metrology='InMetro'
    # Curves is True to print the curves
    Curves=False
    # Fit is True to print the curves
    Fit=False
    # Marsaglia RNG
    RNG='MWC'
    # Seeds
    Seeds=110271,101008
    # Value type : INT32, INT64, FP32, FP64
    ValueType='FP32'
    # Inside based on If
    IfThen=False

    HowToUse='%s -c (Print Curves) -k (Case On IfThen) -d <DeviceId> -g <CUDA/OpenCL> -i <Iterations> -b <BlocksBegin> -e <BlocksEnd> -s <BlocksStep> -f <ThreadsFirst> -l <ThreadsLast> -t <ThreadssTep> -r <RedoToImproveStats> -m <SHR3/CONG/MWC/KISS> -v <INT32/INT64/FP32/FP64>'
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hckg:i:b:e:s:f:l:t:r:d:m:v:",["gpustyle=","iterations=","blocksBegin=","blocksEnd=","blocksStep=","threadsFirst=","threadsLast=","threadssTep=","redo=","device=","marsaglia=","valuetype="])
    except getopt.GetoptError:
        print(HowToUse % sys.argv[0])
        sys.exit(2)

    # List of Devices
    Devices=[]
    Alu={}
        
    for opt, arg in opts:
        if opt == '-h':
            print(HowToUse % sys.argv[0])

            print("\nInformations about devices detected under OpenCL:")
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

                print
            except ImportError:
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
                
        elif opt == '-c':
            Curves=True
        elif opt == '-k':
            IfThen=True
        elif opt in ("-d", "--device"):
            Devices.append(int(arg))
        elif opt in ("-g", "--gpustyle"):
            GpuStyle = arg
        elif opt in ("-m", "--marsaglia"):
            RNG = arg
        elif opt in ("-v", "--valuetype"):
            ValueType = arg
        elif opt in ("-i", "--iterations"):
            Iterations = numpy.uint64(arg)
        elif opt in ("-b", "--blocksbegin"):
            BlocksBegin = int(arg)
            BlocksEnd = BlocksBegin
        elif opt in ("-e", "--blocksend"):
            BlocksEnd = int(arg)
        elif opt in ("-s", "--blocksstep"):
            BlocksStep = int(arg)
        elif opt in ("-f", "--threadsfirst"):
            ThreadsBegin = int(arg)
            ThreadsEnd = ThreadsBegin
        elif opt in ("-l", "--threadslast"):
            ThreadsEnd = int(arg)
        elif opt in ("-t", "--threadsstep"):
            ThreadsStep = int(arg)
        elif opt in ("-r", "--redo"):
            Redo = int(arg)

    print("Devices Identification : %s" % Devices)
    print("GpuStyle used : %s" % GpuStyle)
    print("Iterations : %s" % Iterations)
    print("Number of Blocks on begin : %s" % BlocksBegin)
    print("Number of Blocks on end : %s" % BlocksEnd)
    print("Step on Blocks : %s" % BlocksStep)
    print("Number of Threads on begin : %s" % ThreadsBegin)
    print("Number of Threads on end : %s" % ThreadsEnd)
    print("Step on Threads : %s" % ThreadsStep)
    print("Number of redo : %s" % Redo)
    print("Metrology done out of XPU : %r" % OutMetrology)
    print("Type of Marsaglia RNG used : %s" % RNG)
    print("Type of variable : %s" % ValueType)

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

    print(Devices,Alu)
            
    BlocksList=range(BlocksBegin,BlocksEnd+BlocksStep,BlocksStep)
    ThreadsList=range(ThreadsBegin,ThreadsEnd+ThreadsStep,ThreadsStep)
    
    ExploredJobs=numpy.array([]).astype(numpy.uint32)
    ExploredBlocks=numpy.array([]).astype(numpy.uint32)
    ExploredThreads=numpy.array([]).astype(numpy.uint32)
    avgD=numpy.array([]).astype(numpy.float32)
    medD=numpy.array([]).astype(numpy.float32)
    stdD=numpy.array([]).astype(numpy.float32)
    minD=numpy.array([]).astype(numpy.float32)
    maxD=numpy.array([]).astype(numpy.float32)
    avgR=numpy.array([]).astype(numpy.float32)
    medR=numpy.array([]).astype(numpy.float32)
    stdR=numpy.array([]).astype(numpy.float32)
    minR=numpy.array([]).astype(numpy.float32)
    maxR=numpy.array([]).astype(numpy.float32)

    for Blocks,Threads in itertools.product(BlocksList,ThreadsList):
        
        ExploredJobs=numpy.append(ExploredJobs,Blocks*Threads)
        ExploredBlocks=numpy.append(ExploredBlocks,Blocks)
        ExploredThreads=numpy.append(ExploredThreads,Threads)

        IterationsMP=Iterations/len(Devices)
        if Iterations%len(Devices)!=0:
            IterationsMP+=1

        DurationItem=numpy.array([]).astype(numpy.float32)
        Duration=numpy.array([]).astype(numpy.float32)
        Rate=numpy.array([]).astype(numpy.float32)
        for i in range(Redo):
            MyThreads=[]
            time_start=time.time()

            for Device in Devices:
                DeltaD=Device-min(Devices)+1
                DeltaS=(DeltaD-1)*524287
                InputCL={}
                InputCL['Iterations']=IterationsMP
                InputCL['Steps']=1
                InputCL['Blocks']=Blocks
                InputCL['Threads']=Threads
                InputCL['Device']=Device
                InputCL['RNG']=RNG
                InputCL['Seeds']=numpy.uint32(Seeds[0]*DeltaD+DeltaS),numpy.uint32(Seeds[1]*DeltaD+DeltaS)
                InputCL['ValueType']=ValueType
                InputCL['IfThen']=IfThen
                if GpuStyle=='CUDA':
                    try:
                        MyThread=threadWithReturn(target=MetropolisCuda, args=(InputCL,))
                    except:
                        print("Problem with (%i,%i) // computations on Cuda" % (Blocks,Threads))
                elif GpuStyle=='OpenCL':
                    try:
                        MyThread=threadWithReturn(target=MetropolisOpenCL, args=(InputCL,))
                    except:
                        print("Problem with (%i,%i) // computations on OpenCL" % (Blocks,Threads)            )
                    
                print("Start on #%i device..." % Device)
                MyThread.start()
                MyThreads.append(MyThread)

            NewIterations=0
            Inside=0
            for MyThread in MyThreads:
                OutputCL=MyThread.join()
                NewIterations+=OutputCL['NewIterations']
                Inside+=OutputCL['Inside']

            Duration=numpy.append(Duration,time.time()-time_start)
            Rate=numpy.append(Rate,NewIterations/Duration[-1])
            print("Itops %i\nLogItops %.2f " % (int(Rate[-1]),numpy.log(Rate[-1])/numpy.log(10)))
            print("Pi estimation %.8f" % (4./NewIterations*Inside))
                            
        avgD=numpy.append(avgD,numpy.average(Duration))
        medD=numpy.append(medD,numpy.median(Duration))
        stdD=numpy.append(stdD,numpy.std(Duration))
        minD=numpy.append(minD,numpy.min(Duration))
        maxD=numpy.append(maxD,numpy.max(Duration))
        avgR=numpy.append(avgR,numpy.average(Rate))
        medR=numpy.append(medR,numpy.median(Rate))
        stdR=numpy.append(stdR,numpy.std(Rate))
        minR=numpy.append(minR,numpy.min(Rate))
        maxR=numpy.append(maxR,numpy.max(Rate))

        print("%.2f %.2f %.2f %.2f %.2f %i %i %i %i %i" % (avgD[-1],medD[-1],stdD[-1],minD[-1],maxD[-1],avgR[-1],medR[-1],stdR[-1],minR[-1],maxR[-1]))
        
        numpy.savez("PiThreads_%s_%s_%s_%s_%s_%s_%s_%s_%.8i_Device%i_%s_%s" % (ValueType,RNG,Alu[Devices[0]],GpuStyle,BlocksBegin,BlocksEnd,ThreadsBegin,ThreadsEnd,Iterations,Devices[0],Metrology,gethostname()),(ExploredBlocks,ExploredThreads,avgD,medD,stdD,minD,maxD,avgR,medR,stdR,minR,maxR))
        ToSave=[ ExploredBlocks,ExploredThreads,avgD,medD,stdD,minD,maxD,avgR,medR,stdR,minR,maxR ]
        numpy.savetxt("PiThreads_%s_%s_%s_%s_%s_%s_%s_%i_%.8i_Device%i_%s_%s" % (ValueType,RNG,Alu[Devices[0]],GpuStyle,BlocksBegin,BlocksEnd,ThreadsBegin,ThreadsEnd,Iterations,Devices[0],Metrology,gethostname()),numpy.transpose(ToSave),fmt='%i %i %e %e %e %e %e %i %i %i %i %i')

    if Fit:
        FitAndPrint(ExploredJobs,median,Curves)
