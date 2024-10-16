# Compile all version of Pi_SyCL for nvidia
# Cecill v2 Emmanuel QUEMENER <emmanuel.quemener@gmail.com>

SOURCE=Pi_SyCL.cpp

COMPUTING=INT32 INT64 FP32 FP64
MARSAGLIA=SHR3 CONG MWC KISS

CC=clang++
CFLAGS=-fsycl -fsycl-targets=nvptx64-nvidia-cuda
LIBRARY=

all: $(SOURCE)

	$(foreach TVAR,$(COMPUTING),$(foreach TRND,$(MARSAGLIA),$(CC) $(CFLAGS) -DT$(TVAR) -DT$(TRND) -DLONG -DTIME -o $(<:.cpp=)_$(TVAR)_$(TRND) $< $(LIBRARY); ) )

.PHONY: clean check mrproper

mrproper: 
	rm -rf $(foreach SRC,$(SOURCE),$(foreach TVAR,$(COMPUTING),$(foreach TRND,$(MARSAGLIA),$(SRC:.cpp=)_$(TVAR)_$(TRND) ) ) )

	find . -name "*~" -exec rm {} \;

clean:
	find . -name "*~" -exec rm {} \;

