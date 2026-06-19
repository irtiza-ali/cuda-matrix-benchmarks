CXX =nvcc
CXXFLAGS = -O3 -std=c++14
INCLUDES = -I/usr/include/eigen3
LIBS = -lcublas -lcusolver

all: matrix_benchmark naive_benchmark

matrix_benchmark: matrix_math_opt.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) matrix_math_opt.cu -o matrix_benchmark $(LIBS)

naive_benchmark: matrix_math_naive.cu
	$(CXX) $(CXXFLAGS) matrix_math.cu -o naive_benchmark

clean:
	rm -f matrix_benchmark naive_benchmark results.csv *.png