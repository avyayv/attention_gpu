// attention.h
#ifndef ATTENTION_H
#define ATTENTION_H

#include <vector>
#include <cmath>
#include <algorithm>

float dotProduct(const std::vector<float> &a, const std::vector<float> &b);
std::vector<float> softmax(const std::vector<float> &x);
std::vector<std::vector<float>> scaledDotProductAttention(const std::vector<std::vector<float>> &queries,
                                                          const std::vector<std::vector<float>> &keys,
                                                          const std::vector<std::vector<float>> &values);

#endif // ATTENTION_H
