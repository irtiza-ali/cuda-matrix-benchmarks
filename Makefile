CXX =nvcc
CXXFLAGS = -O3 -std=c++14
INCLUDES = -I/usr/include/eigen3
LIBS = -lcublas -lcusolver

all: matrix_benchmark naive_benchmark

matrix_benchmark: matrix_math_opt.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) matrix_math_opt.cu -o matrix_benchmark $(LIBS)

naive_benchmark: matrix_math_naive.cu
	$(CXX) $(CXXFLAGS) matrix_math_naive.cu -o naive_benchmark
	
fp_benchmark: fp_speed_benchmark.cu
	nvcc -O3 -arch=sm_75 -lcublas fp_speed_benchmark.cu -o fp_speed_benchmark

clean:
	rm -f matrix_benchmark naive_benchmark fp_speed_benchmark results.csv *.png