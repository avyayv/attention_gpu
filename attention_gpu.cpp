#include <cublas_v2.h>
#include <cudnn.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>

#include "helper_cuda.h"

std::vector<std::vector<float>> scaledDotProductAttentionGpu(
    const std::vector<std::vector<float>> &queries,
    const std::vector<std::vector<float>> &keys,
    const std::vector<std::vector<float>> &values,
    cublasHandle_t cublasHandle,
    cudnnHandle_t cudnnHandle)
{

    int batchSize = 1; // Assuming the batch size is 1 for simplicity
    int seqLengthQ = queries.size();
    int seqLengthKV = keys.size();
    int depth = queries[0].size();
    int valueDim = values[0].size();

    float *d_queries;
    float *d_keys;
    float *d_values;
    float *d_tempResult;      // Store intermediate result of QK^T
    float *d_attentionScores; // After softmax
    float *d_output;          // Multiplying the attention score by the value matrix

    size_t bytesQK = seqLengthQ * depth * sizeof(float);
    size_t bytesV = seqLengthKV * valueDim * sizeof(float);
    size_t bytesScores = seqLengthQ * seqLengthKV * sizeof(float);

    CUDA_CALL(cudaMalloc(&d_queries, bytesQK));
    CUDA_CALL(cudaMalloc(&d_keys, bytesQK)); // assume square for simplicity
    CUDA_CALL(cudaMalloc(&d_values, bytesV));
    CUDA_CALL(cudaMalloc(&d_tempResult, bytesScores));
    CUDA_CALL(cudaMalloc(&d_attentionScores, bytesScores));
    CUDA_CALL(cudaMalloc(&d_output, seqLengthQ * valueDim * sizeof(float)));

    // Copy data to device
    cudaMemcpy(d_queries, queries.data(), bytesQK, cudaMemcpyHostToDevice);
    cudaMemcpy(d_keys, keys.data(), bytesQK, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values.data(), bytesV, cudaMemcpyHostToDevice);

    // Perform matrix multiplication QK^T using cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                             seqLengthQ, seqLengthKV, depth,
                             &alpha,
                             d_queries, seqLengthQ,
                             d_keys, seqLengthKV,
                             &beta,
                             d_tempResult, seqLengthQ));

    // Scale the product by sqrt(d_k)
    float scale = 1.0 / std::sqrt(depth);
    CUBLAS_CALL(cublasSscal(cublasHandle, seqLengthQ * seqLengthKV, &scale, d_tempResult, 1));

    // Softmax on the scaled matrix
    cudnnTensorDescriptor_t tensorDesc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&tensorDesc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, seqLengthQ, seqLengthKV, 1));

    const float alphaSoftmax = 1.0;
    const float betaSoftmax = 0.0;
    CUDNN_CALL(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                    &alphaSoftmax, tensorDesc, d_tempResult, &betaSoftmax, tensorDesc, d_attentionScores));

    // Multiply the softmax result with values V
    CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                             seqLengthQ, valueDim, seqLengthKV,
                             &alpha,
                             d_attentionScores, seqLengthQ,
                             d_values, seqLengthKV,
                             &beta,
                             d_output, seqLengthQ));

    // Copy the result back to host
    std::vector<std::vector<float>> output(seqLengthQ, std::vector<float>(valueDim));
    cudaMemcpy(output.data(), d_output, seqLengthQ * valueDim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_queries);
    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_tempResult);
    cudaFree(d_attentionScores);
    cudaFree(d_output);

    cudnnDestroyTensorDescriptor(tensorDesc);

    return output;
}