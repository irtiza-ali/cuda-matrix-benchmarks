%%writefile benchmark_plot.cu
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

// Macro for error checking
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Function to calculate TFLOPS
double calculate_tflops(int N, float ms, int iterations) {
    double operations = 2.0 * (double)N * (double)N * (double)N * (double)iterations;
    double seconds = ms / 1000.0;
    return (operations / seconds) / 1e12; // Convert to TeraFLOPS
}

void run_benchmark(cublasHandle_t handle, int N, std::ofstream& csv) {
    std::cout << "Benchmarking Matrix Size: " << N << " x " << N << "...\n";

    size_t elements = (size_t)N * N;
    int iterations = 10; 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    double tflops;

    // --- FP64 (Double Precision) ---
    double *d_A64, *d_B64, *d_C64;
    CHECK_CUDA(cudaMalloc(&d_A64, elements * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B64, elements * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_C64, elements * sizeof(double)));
    
    double alpha64 = 1.0, beta64 = 0.0;
    CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha64, d_A64, N, d_B64, N, &beta64, d_C64, N));
    
    cudaEventRecord(start);
    for(int i = 0; i < iterations; i++) {
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha64, d_A64, N, d_B64, N, &beta64, d_C64, N);
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&ms, start, stop);
    tflops = calculate_tflops(N, ms, iterations);
    csv << N << ",FP64," << ms / iterations << "," << tflops << "\n";
    std::cout << "  FP64: " << tflops << " TFLOPS\n";
    CHECK_CUDA(cudaFree(d_A64)); CHECK_CUDA(cudaFree(d_B64)); CHECK_CUDA(cudaFree(d_C64));

    // --- FP32 (Single Precision) ---
    float *d_A32, *d_B32, *d_C32;
    CHECK_CUDA(cudaMalloc(&d_A32, elements * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B32, elements * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C32, elements * sizeof(float)));
    
    float alpha32 = 1.0f, beta32 = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha32, d_A32, N, d_B32, N, &beta32, d_C32, N));
    
    cudaEventRecord(start);
    for(int i = 0; i < iterations; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha32, d_A32, N, d_B32, N, &beta32, d_C32, N);
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&ms, start, stop);
    tflops = calculate_tflops(N, ms, iterations);
    csv << N << ",FP32," << ms / iterations << "," << tflops << "\n";
    std::cout << "  FP32: " << tflops << " TFLOPS\n";
    CHECK_CUDA(cudaFree(d_A32)); CHECK_CUDA(cudaFree(d_B32)); CHECK_CUDA(cudaFree(d_C32));

    // --- FP16 (Half Precision with Tensor Cores) ---
    half *d_A16, *d_B16, *d_C16;
    CHECK_CUDA(cudaMalloc(&d_A16, elements * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B16, elements * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C16, elements * sizeof(half)));
    
    half alpha16 = __float2half(1.0f), beta16 = __float2half(0.0f);
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha16, d_A16, CUDA_R_16F, N, d_B16, CUDA_R_16F, N, &beta16, d_C16, CUDA_R_16F, N, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    cudaEventRecord(start);
    for(int i = 0; i < iterations; i++) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha16, d_A16, CUDA_R_16F, N, d_B16, CUDA_R_16F, N, &beta16, d_C16, CUDA_R_16F, N, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&ms, start, stop);
    tflops = calculate_tflops(N, ms, iterations);
    csv << N << ",FP16," << ms / iterations << "," << tflops << "\n";
    std::cout << "  FP16: " << tflops << " TFLOPS (Tensor Cores!)\n\n";
    CHECK_CUDA(cudaFree(d_A16)); CHECK_CUDA(cudaFree(d_B16)); CHECK_CUDA(cudaFree(d_C16));

    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    std::ofstream csv_file("results.csv");
    csv_file << "Size,Precision,Time_ms,TFLOPS\n"; // CSV Header

    std::cout << "Starting Mixed Precision Benchmark...\n\n";
    std::vector<int> sizes = {1024, 2048, 4096, 8192};
    
    for (int N : sizes) {
        run_benchmark(handle, N, csv_file);
    }

    csv_file.close();
    cublasDestroy(handle);
    return 0;
}