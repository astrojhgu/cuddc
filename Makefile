all: test_ddc libcuddc.a

OPT=-O3
CFLAGS = -g $(OPT)

test_ddc.o: test_ddc.cpp
	g++ -c $< -o $@ $(CFLAGS)

ddc_kernel.o: ddc_kernel.cu
	nvcc -c $< -o $@ $(CFLAGS) --cudart=static --cudadevrt=none

test_ddc: test_ddc.o ddc_kernel.o
	nvcc $^ -o $@ $(CFLAGS) --cudart=static --cudadevrt=static

libcuddc.a: ddc_kernel.o
	ar crv $@ $^
	ranlib $@

clean:
	rm -f *.o
