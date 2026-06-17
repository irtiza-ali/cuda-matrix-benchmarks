#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <Eigen/Dense>
#include <cublas_v2.h>
#include <cusolverDn.h>

// Macro for CUDA error checking
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Macro for cuBLAS error checking
#define CHECK_CUBLAS(call) { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Macro for cuSOLVER error checking
#define CHECK_CUSOLVER(call) { \
    cusolverStatus_t err = call; \
    if (err != CUSOLVER_STATUS_SUCCESS) { \
        std::cerr << "cuSOLVER Error at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

using namespace std::chrono;

// Helper to generate a well-conditioned random matrix (Column-major for Eigen/cuBLAS)
Eigen::MatrixXf generateRandomMatrix(int N) {
    Eigen::MatrixXf mat = Eigen::MatrixXf::Random(N, N);
    // Add N to the diagonal to make it diagonally dominant (prevents NaNs during inversion/multiplication)
    for (int i = 0; i < N; ++i) mat(i, i) += N; 
    return mat;
}

// GPU Inversion Helper (Calculates Inverse using LU Factorization)
void gpu_invert(cusolverDnHandle_t cusolver_handle, float* d_A, float* d_invA, int N) {
    int* d_ipiv;
    int* d_info;
    float* d_work;
    int lwork = 0;

    CHECK_CUDA(cudaMalloc(&d_ipiv, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));

    // 1. LU Factorization
    CHECK_CUSOLVER(cusolverDnSgetrf_bufferSize(cusolver_handle, N, N, d_A, N, &lwork));
    CHECK_CUDA(cudaMalloc(&d_work, lwork * sizeof(float)));
    CHECK_CUSOLVER(cusolverDnSgetrf(cusolver_handle, N, N, d_A, N, d_work, d_ipiv, d_info));

    // 2. Solve to get inverse (A * invA = I)
    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(N, N);
    CHECK_CUDA(cudaMemcpy(d_invA, I.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUSOLVER(cusolverDnSgetrs(cusolver_handle, CUBLAS_OP_N, N, N, d_A, N, d_ipiv, d_invA, N, d_info));

    cudaFree(d_ipiv);
    cudaFree(d_info);
    cudaFree(d_work);
}

int main() {
    std::vector<int> sizes = {10, 100, 500, 1000, 2000};
    std::ofstream outfile("results.csv");
    outfile << "Size,CPU_Mul,GPU_Mul,CPU_Inv,GPU_Inv,CPU_Eig,GPU_Eig,CPU_Loss,GPU_Loss\n";

    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUSOLVER(cusolverDnCreate(&cusolver_handle));

    for (int N : sizes) {
        std::cout << "Running Benchmark for N = " << N << "...\n";

        // --- 1. SETUP DATA ---
        Eigen::MatrixXf A = generateRandomMatrix(N);
        Eigen::MatrixXf B = generateRandomMatrix(N);
        Eigen::MatrixXf SymA = A + A.transpose(); // Symmetric matrix for eigenvalues

        // --- 2. MULTIPLICATION BENCHMARK ---
        auto start = high_resolution_clock::now();
        Eigen::MatrixXf C_cpu = A * B;
        auto end = high_resolution_clock::now();
        float cpu_mul_time = duration<float, std::milli>(end - start).count();

        start = high_resolution_clock::now();
        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, N * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, N * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, N * N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_A, A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N));
        CHECK_CUDA(cudaDeviceSynchronize());
        end = high_resolution_clock::now();
        float gpu_mul_time = duration<float, std::milli>(end - start).count();

        // --- 3. INVERSION BENCHMARK ---
        start = high_resolution_clock::now();
        Eigen::MatrixXf Inv_cpu = A.inverse();
        end = high_resolution_clock::now();
        float cpu_inv_time = duration<float, std::milli>(end - start).count();

        start = high_resolution_clock::now();
        float *d_A_copy, *d_invA;
        CHECK_CUDA(cudaMalloc(&d_A_copy, N * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_invA, N * N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_A_copy, A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
        gpu_invert(cusolver_handle, d_A_copy, d_invA, N);
        CHECK_CUDA(cudaDeviceSynchronize());
        end = high_resolution_clock::now();
        float gpu_inv_time = duration<float, std::milli>(end - start).count();

        // --- 4. EIGENVALUE BENCHMARK ---
        start = high_resolution_clock::now();
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigensolver(SymA);
        end = high_resolution_clock::now();
        float cpu_eig_time = duration<float, std::milli>(end - start).count();

        start = high_resolution_clock::now();
        float *d_SymA, *d_W;
        int *d_info;
        float *d_work;
        int lwork = 0;
        CHECK_CUDA(cudaMalloc(&d_SymA, N * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_W, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_SymA, SymA.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUSOLVER(cusolverDnSsyevd_bufferSize(cusolver_handle, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER, N, d_SymA, N, d_W, &lwork));
        CHECK_CUDA(cudaMalloc(&d_work, lwork * sizeof(float)));
        CHECK_CUSOLVER(cusolverDnSsyevd(cusolver_handle, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER, N, d_SymA, N, d_W, d_work, lwork, d_info));
        CHECK_CUDA(cudaDeviceSynchronize());
        end = high_resolution_clock::now();
        float gpu_eig_time = duration<float, std::milli>(end - start).count();

        // --- 5. CUMULATIVE PRECISION LOSS TEST (50 iterations) ---
        // CPU
        Eigen::MatrixXf X_cpu = Eigen::MatrixXf::Identity(N, N);
        for(int k=0; k<50; ++k) {
            X_cpu = X_cpu * A;
            X_cpu = X_cpu * Inv_cpu;
        }
        float cpu_loss = (X_cpu - Eigen::MatrixXf::Identity(N, N)).norm();

        // GPU
        float *d_X, *d_temp;
        CHECK_CUDA(cudaMalloc(&d_X, N * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_temp, N * N * sizeof(float)));
        Eigen::MatrixXf I = Eigen::MatrixXf::Identity(N, N);
        CHECK_CUDA(cudaMemcpy(d_X, I.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
        
        for(int k=0; k<50; ++k) {
            // X = X * A
            CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_X, N, d_A, N, &beta, d_temp, N));
            // X = temp * A_inv
            CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_temp, N, d_invA, N, &beta, d_X, N));
        }
        
        Eigen::MatrixXf X_gpu_res(N, N);
        CHECK_CUDA(cudaMemcpy(X_gpu_res.data(), d_X, N * N * sizeof(float), cudaMemcpyDeviceToHost));
        float gpu_loss = (X_gpu_res - Eigen::MatrixXf::Identity(N, N)).norm();

        // Write Results
        outfile << N << "," << cpu_mul_time << "," << gpu_mul_time << "," 
                << cpu_inv_time << "," << gpu_inv_time << "," 
                << cpu_eig_time << "," << gpu_eig_time << "," 
                << cpu_loss << "," << gpu_loss << "\n";

        // Free Memory
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_A_copy); 
        cudaFree(d_invA); cudaFree(d_SymA); cudaFree(d_W); cudaFree(d_info); 
        cudaFree(d_work); cudaFree(d_X); cudaFree(d_temp);
    }

    cublasDestroy(cublas_handle);
    cusolverDnDestroy(cusolver_handle);
    outfile.close();
    std::cout << "Done! Results saved to results.csv\n";
    return 0;
}