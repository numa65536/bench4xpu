# Makefile designed for Debian Buster

SOURCE=xTRSV.c

CC=gcc
FC=gfortran
CFLAGS=-Wall -O3
LDFLAGS=-lm

CUDAINC?=/usr/include
CUDALIB?=/usr/lib/x86_64-linux-gnu/
CUDASRC?=/usr/share/doc/nvidia-cuda-toolkit/examples/libcublas
THUNKING=fortran_thunking.c
CUDASRCINC=fortran_common.h

GSLINC=/usr/include/gsl

# OPENBLAS=/usr/lib/x86_64-linux-gnu
# ATLAS=/usr/lib/x86_64-linux-gnu

ACML=/opt/acml
ACMLINC=$(ACML)/gfortran64_mp/include
ACMLLIB=$(ACML)/gfortran64_mp/lib

#EXECUTABLE=cblas fblas gsl cublas thunking acml openblas
EXECUTABLE=cblas fblas gsl cublas thunking openblas

# FORMAT define the precision of Floating Point numbers
# FP32 for simple of 32 bits size
# FP64 for double of 64 bits size
#FORMAT=FP64
FORMAT=FP32

#DIRECTIVES=-D$(FORMAT) -DPRINT -DUNIT
#DIRECTIVES=-D$(FORMAT) -DUNIT -DRESULTS -DQUIET
#DIRECTIVES=-DQUIET -DUNIT
DIRECTIVES=-DRESULTS -DQUIET

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
		$(SOURCE) -L$(ACMLLIB) -lacml_mp -lacml_mv \
		-lgomp -lgfortran -lpthread -o $(SOURCE:.c=)_SP_$@

	$(CC) -I$(ACMLINC) $(CFLAGS) $(DIRECTIVES) -DFP64 -DACML $(LDFLAGS) \
		$(SOURCE) -L$(ACMLLIB) -lacml_mp -lacml_mv \
		-lgomp -lgfortran -lpthread -o $(SOURCE:.c=)_DP_$@

fblas: $(SOURCE)

	# ATLAS version
	# $(CC) -L$(ATLAS) $(CFLAGS) $(DIRECTIVES) -DFP32 -DFBLAS $(LDFLAGS) \
	#  	$(SOURCE) -latlas -lf77blas -o $(SOURCE:.c=)_SP_$@

	# $(CC) -L$(ATLAS) $(CFLAGS) $(DIRECTIVES) -DFP64 -DFBLAS $(LDFLAGS) \
	#  	$(SOURCE) -latlas -lf77blas -o $(SOURCE:.c=)_DP_$@

	# OpenBLAS version
	$(CC) $(CFLAGS) $(DIRECTIVES) -DFP32 -DFBLAS $(LDFLAGS) \
	  	$(SOURCE) -lopenblas -lpthread -o $(SOURCE:.c=)_SP_$@

	$(CC) $(CFLAGS) $(DIRECTIVES) -DFP64 -DFBLAS $(LDFLAGS) \
	  	$(SOURCE) -lopenblas -lpthread -o $(SOURCE:.c=)_DP_$@


gsl: $(SOURCE)

	$(CC) -I$(GSLINC) $(CFLAGS) $(DIRECTIVES) -DFP32 -DGSL $(LDFLAGS) \
		$(SOURCE) -lgslcblas -o $(SOURCE:.c=)_SP_$@

	$(CC) -I$(GSLINC) $(CFLAGS) $(DIRECTIVES) -DFP64 -DGSL $(LDFLAGS) \
		$(SOURCE) -lgslcblas -o $(SOURCE:.c=)_DP_$@


cublas: $(SOURCE)

	$(CC) -I$(CUDAINC) -L$(CUDALIB) $(CFLAGS) -DCUBLAS -DFP32 $(LDFLAGS) \
		$(DIRECTIVES) $(SOURCE) -lcublas -o $(SOURCE:.c=)_SP_$@

	$(CC) -I$(CUDAINC) -L$(CUDALIB) $(CFLAGS) -DCUBLAS -DFP64 $(LDFLAGS) \
		$(DIRECTIVES) $(SOURCE) -lcublas -o $(SOURCE:.c=)_DP_$@

thunking: $(SOURCE)

# Copy of source for thunking CUBLAS approach
	cp $(CUDASRC)/$(THUNKING) $(CUDASRC)/$(THUNKING:.c=.h) \
		$(CUDASRC)/$(CUDASRCINC) .

	$(CC) -DCUBLAS_GFORTRAN -I$(CUDAINC) $(CFLAGS) -c $(THUNKING)

	$(CC) -DCUBLAS_GFORTRAN -I$(CUDAINC) -L$(CUDALIB) $(CFLAGS) -DTHUNKING \
		$(LDFLAGS) $(DIRECTIVES) $(SOURCE) -DFP32 \
		$(THUNKING:.c=.o) -lcublas -o $(SOURCE:.c=)_SP_$@

	$(CC) -DCUBLAS_GFORTRAN -I$(CUDAINC) -L$(CUDALIB) $(CFLAGS) -DTHUNKING \
		$(LDFLAGS) $(DIRECTIVES) $(SOURCE) -DFP64 \
		$(THUNKING:.c=.o) -lcublas -o $(SOURCE:.c=)_DP_$@

clean: $(SOURCE)
	find . -name "$(SOURCE:.c=)_*" -exec rm {} \;
	find . -name "$(THUNKING:.c=)*" -exec rm {} \;
	find . -name "*~" -exec rm {} \;
	find . -name "$(CUDASRCINC)" -exec rm {} \;
