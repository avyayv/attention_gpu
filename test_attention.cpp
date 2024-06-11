// test_attention.cpp
#include <iostream>
#include <cassert>
#include "attention.h"
#include "attention_gpu.cpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <algorithm>
#include "helper_cuda.h"

void testDotProduct()
{
    std::vector<float> a = {1.0, 2.0, 3.0};
    std::vector<float> b = {4.0, 5.0, 6.0};
    assert(dotProduct(a, b) == 32.0);
    std::cout << "dotProduct test passed" << std::endl;
}

void testSoftmax()
{
    std::vector<float> x = {1.0, 2.0, 3.0};
    std::vector<float> result = softmax(x);
    assert(result.size() == x.size());
    std::cout << "softmax test passed" << std::endl;
}

void testScaledDotProductAttention()
{
    std::vector<std::vector<float>> queries = {{1.0, 0.0, 1.0}, {0.0, 1.0, 0.0}};
    std::vector<std::vector<float>> keys = {{1.0, 0.0, 1.0}, {0.0, 1.0, 0.0}};
    std::vector<std::vector<float>> values = {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}};

    std::vector<std::vector<float>> output = scaledDotProductAttention(queries, keys, values);
    assert(output.size() == queries.size());
    assert(output[0].size() == values[0].size());
    std::cout << "scaledDotProductAttention test passed" << std::endl;
}

void testScaledDotProductAttentionGpu()
{
    // Prepare sample data for the test
    std::vector<std::vector<float>> queries = {{1.0, 0.0, 1.0}, {0.0, 1.0, 0.0}};
    std::vector<std::vector<float>> keys = {{1.0, 0.0, 1.0}, {0.0, 1.0, 0.0}};
    std::vector<std::vector<float>> values = {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}};

    // Initialize cuBLAS and cuDNN handles
    cublasHandle_t cublasHandle;
    cudnnHandle_t cudnnHandle;
    cublasCreate(&cublasHandle);
    cudnnCreate(&cudnnHandle);

    // Execute the GPU-accelerated attention function
    std::vector<std::vector<float>> output = scaledDotProductAttentionGpu(queries, keys, values, cublasHandle, cudnnHandle);

    // Verify that the output dimensions are correct
    assert(output.size() == queries.size() && "Output size should match number of queries");
    for (size_t i = 0; i < output.size(); ++i)
    {
        assert(output[i].size() == values[0].size() && "Inner dimension of output should match value dimension");
    }

    // Log successful test completion
    std::cout << "GPU-based scaledDotProductAttention test passed" << std::endl;

    // Cleanup resources
    cublasDestroy(cublasHandle);
    cudnnDestroy(cudnnHandle);
}

int main()
{
    testDotProduct();
    testSoftmax();
    testScaledDotProductAttention();
    testScaledDotProductAttentionGpu();
    return 0;
}
