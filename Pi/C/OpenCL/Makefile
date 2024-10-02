SOURCE=PiOpenCL.c PiOpenCL_CONG.c PiOpenCL_SHR3.c PiOpenCL_MWC.c PiOpenCL_KISS.c

CC=gcc
CFLAGS=-Wall -Wno-unused-variable -Wno-sequence-point -O3 -std=c99
LIBRARY=-lm -lOpenCL

all: $(SOURCE)

	$(foreach SRC,$(SOURCE),$(CC) $(CFLAGS) -o $(SRC:.c=) $(SRC) $(LIBRARY); )

.PHONY: clean check mrproper

mrproper: 
	rm -rf $(foreach SRC,$(SOURCE),$(SRC:.c=) )
	find . -name "*~" -exec rm {} \;

clean:
	find . -name "*~" -exec rm {} \;
