#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <random>

using namespace std;

// ==========================================
// Error Checking Macro
// ==========================================
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// ==========================================
// CPU Naive Implementations
// ==========================================

void cpu_matrix_mult(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void cpu_matrix_inv(const float* A, float* Inv, int N) {
    vector<float> aug(N * 2 * N, 0.0f);
    // Initialize augmented matrix [A | I]
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            aug[i * 2 * N + j] = A[i * N + j];
        }
        aug[i * 2 * N + (i + N)] = 1.0f;
    }

    // Gauss-Jordan Elimination
    for (int i = 0; i < N; ++i) {
        float pivot = aug[i * 2 * N + i];
        // Normalize pivot row
        for (int j = 0; j < 2 * N; ++j) {
            aug[i * 2 * N + j] /= pivot;
        }
        // Eliminate other rows
        for (int k = 0; k < N; ++k) {
            if (k != i) {
                float factor = aug[k * 2 * N + i];
                for (int j = 0; j < 2 * N; ++j) {
                    aug[k * 2 * N + j] -= factor * aug[i * 2 * N + j];
                }
            }
        }
    }

    // Extract Inverse
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            Inv[i * N + j] = aug[i * 2 * N + (j + N)];
        }
    }
}

float cpu_power_iteration(const float* A, int N, int iters = 50) {
    vector<float> v(N, 1.0f);
    vector<float> v_next(N, 0.0f);

    for (int iter = 0; iter < iters; ++iter) {
        // v_next = A * v
        for (int i = 0; i < N; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < N; ++j) {
                sum += A[i * N + j] * v[j];
            }
            v_next[i] = sum;
        }
        // Norm
        float norm_sq = 0.0f;
        for (int i = 0; i < N; ++i) norm_sq += v_next[i] * v_next[i];
        float norm = sqrt(norm_sq);
        // Normalize
        for (int i = 0; i < N; ++i) v[i] = v_next[i] / norm;
    }
    
    // Rayleigh quotient for eigenvalue
    float numerator = 0.0f, denominator = 0.0f;
    for (int i = 0; i < N; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) sum += A[i * N + j] * v[j];
        numerator += v[i] * sum;
        denominator += v[i] * v[i];
    }
    return numerator / denominator;
}

// ==========================================
// GPU Naive Implementations (Kernels)
// ==========================================

__global__ void gpu_matrix_mult_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Gauss-Jordan kernels
__global__ void gpu_normalize_row_kernel(float* aug, int N, int pivot_row, float pivot_val) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < 2 * N) {
        aug[pivot_row * 2 * N + col] /= pivot_val;
    }
}

__global__ void gpu_eliminate_rows_kernel(float* aug, int N, int pivot_row) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && row != pivot_row && col < 2 * N) {
        float factor = aug[row * 2 * N + pivot_row];
        // Syncthreads isn't enough across blocks. We must compute factor before modifying, 
        // but wait, factor depends on the column being updated? No, factor is at col=pivot_row.
        // To strictly avoid race conditions without using shared memory in a naive way:
        // We do this element-wise. BUT factor must be read before the thread overwrites it!
        // So we read it locally first.
        aug[row * 2 * N + col] -= factor * aug[pivot_row * 2 * N + col];
    }
}

// Power iteration kernels
__global__ void gpu_mat_vec_mult(const float* A, const float* v, float* v_out, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) sum += A[row * N + j] * v[j];
        v_out[row] = sum;
    }
}

__global__ void gpu_calc_norm_sq(const float* v, float* norm_sq, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        atomicAdd(norm_sq, v[i] * v[i]);
    }
}

__global__ void gpu_normalize_vec(float* v, float norm, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        v[i] /= norm;
    }
}

// ==========================================
// Helper Functions
// ==========================================

float calc_precision_error(const float* A, const float* A_inv, int N) {
    vector<float> I_approx(N * N, 0.0f);
    cpu_matrix_mult(A, A_inv, I_approx.data(), N);
    
    float error = 0.0f;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            float diff = I_approx[i * N + j] - expected;
            error += diff * diff;
        }
    }
    return sqrt(error); // Frobenius norm
}

void init_matrix(float* A, int N) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.1f, 1.0f);
    for (int i = 0; i < N * N; ++i) A[i] = dis(gen);
    
    // Make strictly diagonally dominant to guarantee invertible / no zero pivots
    for (int i = 0; i < N; ++i) A[i * N + i] += N; 
}

// ==========================================
// Main Benchmark Loop
// ==========================================
int main() {
    ofstream out("results.csv");
    out << "N,CPU_Mult_ms,GPU_Mult_ms,CPU_Inv_ms,GPU_Inv_ms,CPU_Eigen_ms,GPU_Eigen_ms,CPU_Error,GPU_Error\n";

    cout << "Starting Benchmarks...\n";

    for (int N = 2; N <= 1024; N *= 2) {
        cout << "Testing N = " << N << "..." << endl;
        size_t bytes = N * N * sizeof(float);

        vector<float> h_A(N * N), h_B(N * N), h_C_cpu(N * N), h_Inv_cpu(N * N), h_Inv_gpu(N * N);
        init_matrix(h_A.data(), N);
        init_matrix(h_B.data(), N);

        // -----------------------------------------------------
        // MULTIPLICATION
        // -----------------------------------------------------
        auto start = chrono::high_resolution_clock::now();
        cpu_matrix_mult(h_A.data(), h_B.data(), h_C_cpu.data(), N);
        auto end = chrono::high_resolution_clock::now();
        float cpu_mult_ms = chrono::duration<float, milli>(end - start).count();

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
        cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

        dim3 threads(16, 16);
        dim3 blocks((N + 15) / 16, (N + 15) / 16);
        
        cudaEvent_t e_start, e_stop;
        cudaEventCreate(&e_start); cudaEventCreate(&e_stop);

        cudaEventRecord(e_start);
        gpu_matrix_mult_kernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
        cudaEventRecord(e_stop);
        cudaEventSynchronize(e_stop);
        
        float gpu_mult_ms = 0;
        cudaEventElapsedTime(&gpu_mult_ms, e_start, e_stop);

        // -----------------------------------------------------
        // INVERSION
        // -----------------------------------------------------
        start = chrono::high_resolution_clock::now();
        cpu_matrix_inv(h_A.data(), h_Inv_cpu.data(), N);
        end = chrono::high_resolution_clock::now();
        float cpu_inv_ms = chrono::duration<float, milli>(end - start).count();

        float *d_aug;
        cudaMalloc(&d_aug, N * 2 * N * sizeof(float));
        vector<float> h_aug(N * 2 * N, 0.0f);
        for(int i=0; i<N; ++i) {
            for(int j=0; j<N; ++j) h_aug[i * 2 * N + j] = h_A[i * N + j];
            h_aug[i * 2 * N + (i + N)] = 1.0f;
        }
        cudaMemcpy(d_aug, h_aug.data(), N * 2 * N * sizeof(float), cudaMemcpyHostToDevice);

        dim3 inv_threads(16, 16);
        dim3 inv_blocks((2 * N + 15) / 16, (N + 15) / 16);
        int threads1D = 256;
        int blocks1D = (2 * N + threads1D - 1) / threads1D;

        cudaEventRecord(e_start);
        for (int i = 0; i < N; ++i) {
            float pivot_val;
            cudaMemcpy(&pivot_val, &d_aug[i * 2 * N + i], sizeof(float), cudaMemcpyDeviceToHost);
            gpu_normalize_row_kernel<<<blocks1D, threads1D>>>(d_aug, N, i, pivot_val);
            gpu_eliminate_rows_kernel<<<inv_blocks, inv_threads>>>(d_aug, N, i);
        }
        cudaEventRecord(e_stop);
        cudaEventSynchronize(e_stop);
        
        float gpu_inv_ms = 0;
        cudaEventElapsedTime(&gpu_inv_ms, e_start, e_stop);

        cudaMemcpy(h_aug.data(), d_aug, N * 2 * N * sizeof(float), cudaMemcpyDeviceToHost);
        for(int i=0; i<N; ++i)
            for(int j=0; j<N; ++j)
                h_Inv_gpu[i * N + j] = h_aug[i * 2 * N + (j + N)];

        // -----------------------------------------------------
        // EIGENVALUE (Power Iteration)
        // -----------------------------------------------------
        start = chrono::high_resolution_clock::now();
        cpu_power_iteration(h_A.data(), N, 50);
        end = chrono::high_resolution_clock::now();
        float cpu_eigen_ms = chrono::duration<float, milli>(end - start).count();

        float *d_v, *d_v_next, *d_norm_sq;
        cudaMalloc(&d_v, N * sizeof(float));
        cudaMalloc(&d_v_next, N * sizeof(float));
        cudaMalloc(&d_norm_sq, sizeof(float));
        vector<float> h_v(N, 1.0f);
        cudaMemcpy(d_v, h_v.data(), N * sizeof(float), cudaMemcpyHostToDevice);

        int vec_blocks = (N + 255) / 256;

        cudaEventRecord(e_start);
        for(int iter = 0; iter < 50; ++iter) {
            gpu_mat_vec_mult<<<vec_blocks, 256>>>(d_A, d_v, d_v_next, N);
            
            float zero = 0.0f;
            cudaMemcpy(d_norm_sq, &zero, sizeof(float), cudaMemcpyHostToDevice);
            gpu_calc_norm_sq<<<vec_blocks, 256>>>(d_v_next, d_norm_sq, N);
            
            float norm_sq;
            cudaMemcpy(&norm_sq, d_norm_sq, sizeof(float), cudaMemcpyDeviceToHost);
            float norm = sqrt(norm_sq);
            
            gpu_normalize_vec<<<vec_blocks, 256>>>(d_v_next, norm, N);
            
            // Swap pointers
            float* temp = d_v; d_v = d_v_next; d_v_next = temp;
        }
        cudaEventRecord(e_stop);
        cudaEventSynchronize(e_stop);
        float gpu_eigen_ms = 0;
        cudaEventElapsedTime(&gpu_eigen_ms, e_start, e_stop);

        // -----------------------------------------------------
        // PRECISION LOSS TESTING
        // -----------------------------------------------------
        float cpu_error = calc_precision_error(h_A.data(), h_Inv_cpu.data(), N);
        float gpu_error = calc_precision_error(h_A.data(), h_Inv_gpu.data(), N);

        // Output and cleanup
        out << N << "," << cpu_mult_ms << "," << gpu_mult_ms << "," 
            << cpu_inv_ms << "," << gpu_inv_ms << "," 
            << cpu_eigen_ms << "," << gpu_eigen_ms << "," 
            << cpu_error << "," << gpu_error << "\n";

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); 
        cudaFree(d_aug); cudaFree(d_v); cudaFree(d_v_next); cudaFree(d_norm_sq);
    }
    
    out.close();
    cout << "Benchmarks completed. Results saved to results.csv\n";
    return 0;
}