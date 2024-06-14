// test_attention.cpp
#include "attention.h"
#include "attention_gpu.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"

#include <cassert>
#include <ctime>
#include <cmath>
#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cudnn.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

cudaEvent_t start;
cudaEvent_t stop;

#define START_TIMER() {                         \
      gpuErrchk(cudaEventCreate(&start));       \
      gpuErrchk(cudaEventCreate(&stop));        \
      gpuErrchk(cudaEventRecord(start));        \
    }

#define STOP_RECORD_TIMER(name) {                           \
      gpuErrchk(cudaEventRecord(stop));                     \
      gpuErrchk(cudaEventSynchronize(stop));                \
      gpuErrchk(cudaEventElapsedTime(&name, start, stop));  \
      gpuErrchk(cudaEventDestroy(start));                   \
      gpuErrchk(cudaEventDestroy(stop));                    \
    }

std::vector<std::vector<float>> generateRandomMatrix(int n, int dim) {
    std::vector<std::vector<float>> matrix(n, std::vector<float>(dim));

    // Fill the matrix with random numbers
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < dim; ++j) {
            matrix[i][j] = static_cast<float>(std::rand()) / RAND_MAX; // Random number between 0 and 1
        }
    }

    return matrix;
}

double calculateFrobeniusNorm(const std::vector<std::vector<float>>& matrix) {
    float norm = 0.0;
    for(const auto& row : matrix) {
        for(float val : row) {
            norm += val * val;
        }
    }
    return std::sqrt(norm);
}

void testScaledDotProductAttention()
{
    // Initialize cuBLAS and cuDNN handles
    cublasHandle_t cublasHandle;
    cudnnHandle_t cudnnHandle;
    cublasCreate(&cublasHandle);
    cudnnCreate(&cudnnHandle);

    for (int n = 2; n < 3e3; n *= 2) {
        std::cout << "Sequence Length: " << n << std::endl;
        for (int i = 0; i < 25; ++i) {
            std::cout << "-";
        }
        std::cout << "\n" << std::endl;

        std::vector<std::vector<float>> queries = generateRandomMatrix(n, 128);
        std::vector<std::vector<float>> keys = generateRandomMatrix(n, 128);
        std::vector<std::vector<float>> values = generateRandomMatrix(n, 128);
        
        float cpu_time_ms = -1;
        float gpu_time_ms = -1;

        START_TIMER();
        
        std::vector<std::vector<float>> cpu_output = scaledDotProductAttention(queries, keys, values);

        STOP_RECORD_TIMER(cpu_time_ms);

        std::cout << "CPU-based scaledDotProductAttention runtime: " << cpu_time_ms << " ms" << std::endl;

        START_TIMER();

        std::vector<std::vector<float>> gpu_output = scaledDotProductAttentionGpu(queries, keys, values, cublasHandle, cudnnHandle);
        
        STOP_RECORD_TIMER(gpu_time_ms);

        std::cout << "GPU-based scaledDotProductAttention runtime: " << gpu_time_ms << " ms" << std::endl;

        if (fabs(calculateFrobeniusNorm(cpu_output) - calculateFrobeniusNorm(gpu_output)) < 0.1) {
            std::cout << "CPU and GPU outputs match" << std::endl;
        }

            std::cout << "GPU speed up factor: " << std::round(cpu_time_ms/gpu_time_ms) << "x\n" << std::endl;
    }

    // Cleanup resources
    cublasDestroy(cublasHandle);
    cudnnDestroy(cudnnHandle);
}

int main()
{
    testScaledDotProductAttention();
    return 0;
}
