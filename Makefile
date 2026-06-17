CXX = nvcc
CXXFLAGS = -O3 -std=c++14
INCLUDES = -I/usr/include/eigen3
LIBS = -lcublas -lcusolver

all: matrix_benchmark

matrix_benchmark: main.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.cu -o matrix_benchmark $(LIBS)

clean:
	rm -f matrix_benchmark results.csv