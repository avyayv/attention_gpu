// attention.cpp
#include "attention.h"

float dotProduct(const std::vector<float> &a, const std::vector<float> &b)
{
    float result = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        result += a[i] * b[i];
    }

    return result;
}

std::vector<float> softmax(const std::vector<float> &x)
{
    std::vector<float> result(x.size());
    float maxVal = *std::max_element(x.begin(), x.end());
    float sum = 0.0;

    for (size_t i = 0; i < x.size(); ++i)
    {
        result[i] = std::exp(x[i] - maxVal);
        sum += result[i];
    }

    for (size_t i = 0; i < x.size(); ++i)
    {
        result[i] /= sum;
    }

    return result;
}

std::vector<std::vector<float>> scaledDotProductAttention(const std::vector<std::vector<float>> &queries,
                                                          const std::vector<std::vector<float>> &keys,
                                                          const std::vector<std::vector<float>> &values)
{
    size_t seqLength = queries.size();
    size_t dim = queries[0].size();

    std::vector<std::vector<float>> attentionScores(seqLength, std::vector<float>(seqLength, 0.0));

    // Compute attention scores
    for (size_t i = 0; i < seqLength; ++i)
    {
        for (size_t j = 0; j < seqLength; ++j)
        {
            attentionScores[i][j] = dotProduct(queries[i], keys[j]) / std::sqrt(dim);
        }
    }

    // Apply softmax to attention scores
    for (size_t i = 0; i < seqLength; ++i)
    {
        attentionScores[i] = softmax(attentionScores[i]);
    }

    // Compute the output by multiplying the attention scores with the values
    std::vector<std::vector<float>> output(seqLength, std::vector<float>(values[0].size(), 0.0));
    for (size_t i = 0; i < seqLength; ++i)
    {
        for (size_t j = 0; j < values[0].size(); ++j)
        {
            for (size_t k = 0; k < seqLength; ++k)
            {
                output[i][j] += attentionScores[i][k] * values[k][j];
            }
        }
    }

    return output;
}
