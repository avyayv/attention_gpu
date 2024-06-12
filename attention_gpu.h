// attention.h
#ifndef ATTENTION_GPU_H
#define ATTENTION_GPU_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cudnn.h>

std::vector<std::vector<float>> scaledDotProductAttentionGpu(
    const std::vector<std::vector<float>> &queries,
    const std::vector<std::vector<float>> &keys,
    const std::vector<std::vector<float>> &values,
    cublasHandle_t cublasHandle,
    cudnnHandle_t cudnnHandle);

#endif // ATTENTION_GPU_H
