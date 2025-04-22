all: test_ddc libcuddc.a

OPT=-O3

test_ddc.o: test_ddc.cpp
	g++ -c $< -o $@ $(OPT)

ddc_kernel.o: ddc_kernel.cu
	nvcc -c $< -o $@ $(OPT) --cudart=static --cudadevrt=none

test_ddc: test_ddc.o ddc_kernel.o
	nvcc $^ -o $@ $(OPT) --cudart=static --cudadevrt=static

libcuddc.a: ddc_kernel.o
	ar crv $@ $^
	ranlib $@

clean:
	rm -f *.o
