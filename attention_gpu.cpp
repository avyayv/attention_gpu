#include <cublas_v2.h>
#include <cudnn.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>

#include "helper_cuda.h"

float* flatten(const std::vector<std::vector<float>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    float* flat = new float[rows * cols];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            flat[j * rows + i] = matrix[i][j];
        }
    }
    return flat;
}

std::vector<std::vector<float>> scaledDotProductAttentionGpu(
    const std::vector<std::vector<float>> &queries,
    const std::vector<std::vector<float>> &keys,
    const std::vector<std::vector<float>> &values,
    cublasHandle_t cublasHandle,
    cudnnHandle_t cudnnHandle)
{
    int m = queries.size();       // number of rows in queries and the output matrix
    int k = queries[0].size();    // number of columns in queries and rows in keys^T
    int n = keys.size();          // number of columns in keys and the output matrix
    int v_dim = values[0].size(); // number of columns in values

    // Flatten the matrices for GPU computation, assuming these functions are defined to ensure contiguous storage
    float* d_queries = flatten(queries);
    float* d_keys = flatten(keys);
    float *d_values = flatten(values);

    // Allocate memory on the GPU
    float *d_q, *d_k, *d_v, *d_result, *d_attention_scores;
    cudaMalloc(&d_q, m * k * sizeof(float));
    cudaMalloc(&d_k, n * k * sizeof(float));
    cudaMalloc(&d_v, n * v_dim * sizeof(float));
    cudaMalloc(&d_result, m * v_dim * sizeof(float));
    cudaMalloc(&d_attention_scores, n * m * sizeof(float));

    // Copy matrices to device
    cudaMemcpy(d_q, d_queries, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, d_keys, n * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, d_values, n * v_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Perform matrix multiplication using cuBLAS: d_attention_scores = d_queries * d_keys^T
    const float alpha = 1.0f / sqrtf(k);
    const float beta = 0.0f;
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                m, n, k,
                &alpha,
                d_q, m,
                d_k, n,
                &beta,
                d_attention_scores, m);
    
    // Apply softmax to the attention scores
    cudnnTensorDescriptor_t tensorDesc;
    cudnnCreateTensorDescriptor(&tensorDesc);
    cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, m, n, 1, 1);

    float *d_softmax_output;
    cudaMalloc(&d_softmax_output, m * n * sizeof(float));
    float softmax_alpha = 1.0f, softmax_beta = 0.0f;
    cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                        &softmax_alpha, tensorDesc, d_attention_scores, &softmax_beta, tensorDesc, d_softmax_output);

    std::vector<float> result(m * n);
    cudaMemcpy(result.data(), d_softmax_output, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    for (float elem : result) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;

    std::vector<float> v_result(n * v_dim);
    cudaMemcpy(v_result.data(), d_v, n * v_dim * sizeof(float), cudaMemcpyDeviceToHost);

    for (float elem : v_result) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;

    const float values_alpha = 1.0f;
    const float values_beta = 0.0f;
    // Multiply softmax outputs with values
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, k, n, 
                &values_alpha,
                d_softmax_output, n,
                d_v, n,
                &values_beta,
                d_result, n); 

    // Copy result back to host
    std::vector<float> flat_result(m * v_dim);
    cudaMemcpy(flat_result.data(), d_result, m * v_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Reshape flat_result into a 2D vector
    std::vector<std::vector<float>> host_result(n, std::vector<float>(v_dim));
    for (int i = 0; i < n; ++i)
    {
        std::copy(flat_result.begin() + i * v_dim, flat_result.begin() + (i + 1) * v_dim, host_result[i].begin());
    }

    // Free GPU memory
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_result);
    cudaFree(d_attention_scores);
    cudaFree(d_softmax_output);
    cudnnDestroyTensorDescriptor(tensorDesc);

    return host_result;
}