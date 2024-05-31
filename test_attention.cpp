// test_attention.cpp
#include <iostream>
#include <cassert>
#include "attention.h"

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

int main()
{
    testDotProduct();
    testSoftmax();
    testScaledDotProductAttention();
    return 0;
}
