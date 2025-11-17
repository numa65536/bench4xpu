# Makefile designed for Debian Buster

SOURCE=xGEMM.c

CC=gcc
FC=gfortran
CFLAGS=-Wall -O3 -ffast-math
LDFLAGS=-lm

CLBLASINC=/usr/include
CLBLASLIB=/usr/lib/x86_64-linux-gnu

CLBLASTINC=/usr/include
CLBLASTLIB=/usr/lib/x86_64-linux-gnu

CUDALIB?=/usr/lib/x86_64-linux-gnu/
CUDAINC?=/usr/include
CUDASRC?=/usr/share/doc/nvidia-cuda-toolkit/examples/libcublas
THUNKING=fortran_thunking.c
CUDASRCINC=fortran_common.h

GSLINC=/usr/include/gsl

# OPENBLAS=/usr/lib/x86_64-linux-gnu
# ATLAS=/usr/lib/x86_64-linux-gnu

ACML=/opt/acml
ACMLINC=$(ACML)/gfortran64_mp/include
ACMLLIB=$(ACML)/gfortran64_mp/lib

#EXECUTABLE=gsl cblas fblas openblas clblas cublas acml thunking
EXECUTABLE=gsl fblas openblas clblas cublas thunking clblast

# FORMAT define the precision of Floating Point numbers
# FP32 for simple of 32 bits size
# FP64 for double of 64 bits size
FORMAT=FP64
#FORMAT=FP32

#DIRECTIVES=-D$(FORMAT) -DPRINT -DUNIT
#DIRECTIVES=-D$(FORMAT) -DUNIT -DRESULTS -DQUIET
DIRECTIVES=-DUNIT -DQUIET

all: $(EXECUTABLE)

cblas: $(SOURCE)

	$(CC) $(CFLAGS) $(DIRECTIVES) -DFP32 -DCBLAS $(LDFLAGS) \
		$(SOURCE) -latlas -lcblas -o $(SOURCE:.c=)_SP_$@

	$(CC) $(CFLAGS) $(DIRECTIVES) -DFP64 -DCBLAS $(LDFLAGS) \
		$(SOURCE) -latlas -lcblas -o $(SOURCE:.c=)_DP_$@

openblas: $(SOURCE)

	$(CC) $(CFLAGS) $(DIRECTIVES) -DFP32 -DCBLAS $(LDFLAGS) \
		$(SOURCE) -lopenblas -lpthread -o $(SOURCE:.c=)_SP_$@

	$(CC) $(CFLAGS) $(DIRECTIVES) -DFP64 -DCBLAS $(LDFLAGS) \
		$(SOURCE) -lopenblas -lpthread -o $(SOURCE:.c=)_DP_$@

acml: $(SOURCE)

	$(CC) -I$(ACMLINC) $(CFLAGS) $(DIRECTIVES) -DFP32 -DACML $(LDFLAGS) \
		$(SOURCE) -L$(ACMLLIB) -lacml_mp \
		-lgomp -lgfortran -lpthread -o $(SOURCE:.c=)_SP_$@

	$(CC) -I$(ACMLINC) $(CFLAGS) $(DIRECTIVES) -DFP64 -DACML $(LDFLAGS) \
		$(SOURCE) -L$(ACMLLIB) -lacml_mp \
		-lgomp -lgfortran -lpthread -o $(SOURCE:.c=)_DP_$@

fblas: $(SOURCE)

	# ATLAS version
	# $(CC) -L$(ATLAS) $(CFLAGS) $(DIRECTIVES) -DFP32 -DFBLAS $(LDFLAGS) \
	# 	$(SOURCE) -latlas -lf77blas -o $(SOURCE:.c=)_SP_$@

	# $(CC) -L$(ATLAS) $(CFLAGS) $(DIRECTIVES) -DFP64 -DFBLAS $(LDFLAGS) \
	# 	$(SOURCE) -latlas -lf77blas -o $(SOURCE:.c=)_DP_$@

	# OpenBLAS version
	$(CC) $(CFLAGS) $(DIRECTIVES) -DFP32 -DFBLAS $(LDFLAGS) \
		$(SOURCE) -lopenblas -lpthread -o $(SOURCE:.c=)_SP_$@

	$(CC) $(CFLAGS) $(DIRECTIVES) -DFP64 -DFBLAS $(LDFLAGS) \
		$(SOURCE) -lopenblas -lpthread -o $(SOURCE:.c=)_DP_$@


gsl: $(SOURCE)

	$(CC) -I$(GSLINC) $(CFLAGS) $(DIRECTIVES) -DFP32 \
		-DGSL $(LDFLAGS) \
		$(SOURCE) -lgslcblas -o $(SOURCE:.c=)_SP_$@

	$(CC) -I$(GSLINC) $(CFLAGS) $(DIRECTIVES) -DFP64 \
		-DGSL $(LDFLAGS) \
		$(SOURCE) -lgslcblas -o $(SOURCE:.c=)_DP_$@

clblas: $(SOURCE)

	$(CC) -I$(CLBLASINC) -L$(CLBLASLIB) $(CFLAGS) -DFP32 \
		-DCLBLAS $(LDFLAGS) \
		$(DIRECTIVES) $(SOURCE) -lclBLAS -lOpenCL -o $(SOURCE:.c=)_SP_$@

	$(CC) -I$(CLBLASINC) -L$(CLBLASLIB) $(CFLAGS) -DFP64 \
		-DCLBLAS $(LDFLAGS) \
		$(DIRECTIVES) $(SOURCE) -lclBLAS -lOpenCL -o $(SOURCE:.c=)_DP_$@

clblast: $(SOURCE)

	$(CC) -I$(CLBLASINC) -L$(CLBLASLIB) $(CFLAGS) -DFP32 \
		-DCLBLAST $(LDFLAGS) \
		$(DIRECTIVES) $(SOURCE) -lclblast -lOpenCL -o $(SOURCE:.c=)_SP_$@

	$(CC) -I$(CLBLASINC) -L$(CLBLASLIB) $(CFLAGS) -DFP64 \
		-DCLBLAST $(LDFLAGS) \
		$(DIRECTIVES) $(SOURCE) -lclblast -lOpenCL -o $(SOURCE:.c=)_DP_$@

cublas: $(SOURCE)

	$(CC) -I$(CUDAINC) -L$(CUDALIB) $(CFLAGS) -DFP32 \
		-DCUBLAS $(LDFLAGS) \
		$(DIRECTIVES) $(SOURCE) -lcublas -o $(SOURCE:.c=)_SP_$@

	$(CC) -I$(CUDAINC) -L$(CUDALIB) $(CFLAGS) -DFP64 \
		-DCUBLAS $(LDFLAGS) \
		$(DIRECTIVES) $(SOURCE) -lcublas -o $(SOURCE:.c=)_DP_$@

thunking: $(SOURCE)

	# Copy of source for thunking CUBLAS approach
	cp $(CUDASRC)/$(THUNKING) $(CUDASRC)/$(THUNKING:.c=.h) \
		$(CUDASRC)/$(CUDASRCINC) .
	# Patch Thunking prototypes to compile on Debian Lenny
	$(CC) -DCUBLAS_GFORTRAN -I$(CUDAINC) $(CFLAGS) -c $(THUNKING)

	$(CC) -DCUBLAS_GFORTRAN \
		-I$(CUDAINC) -L$(CUDALIB) $(CFLAGS) -DFP32 -DTHUNKING \
		$(LDFLAGS) $(DIRECTIVES) $(SOURCE) \
		$(THUNKING:.c=.o) -lcublas -o $(SOURCE:.c=)_SP_$@

	$(CC) -DCUBLAS_GFORTRAN \
		-I$(CUDAINC) -L$(CUDALIB) $(CFLAGS) -DFP64 -DTHUNKING \
		$(LDFLAGS) $(DIRECTIVES) $(SOURCE) \
		$(THUNKING:.c=.o) -lcublas -o $(SOURCE:.c=)_DP_$@

clean: $(SOURCE)
	find . -name "$(SOURCE:.c=)_*" -exec rm {} \;
	find . -name "$(THUNKING:.c=)*" -exec rm {} \;
	find . -name "*~" -exec rm {} \;
	find . -name "$(CUDASRCINC)" -exec rm {} \;

check: $(EXECUTABLE)

	$(SOURCE:.c=)_SP_$(EXECUTABLE) 2 1

