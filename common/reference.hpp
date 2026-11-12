#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

// CPU-only mathematical oracles. No GPU solutions live in this file.
namespace reference {
inline std::vector<double> softmax(const std::vector<double>& x) {
    if (x.empty()) throw std::invalid_argument("softmax requires nonempty input");
    double maximum = *std::max_element(x.begin(), x.end()), sum = 0;
    std::vector<double> p(x.size());
    for (size_t i = 0; i < x.size(); ++i) sum += p[i] = std::exp(x[i] - maximum);
    for (double& v : p) v /= sum;
    return p;
}

template <class Query, class Key, class Value>
std::vector<double> attention_row(int tokens, int dim, Query q, Key k, Value v) {
    std::vector<double> scores(tokens), result(dim, 0);
    const double scale = 1.0 / std::sqrt(double(dim));
    for (int s = 0; s < tokens; ++s) {
        double dot = 0;
        for (int j = 0; j < dim; ++j) dot += double(q(j)) * double(k(s, j));
        scores[s] = dot * scale;
    }
    auto weights = softmax(scores);
    for (int s = 0; s < tokens; ++s)
        for (int j = 0; j < dim; ++j) result[j] += weights[s] * double(v(s, j));
    return result;
}

inline void attention_backward(const std::vector<float>& q, const std::vector<float>& k,
                               const std::vector<float>& v, const std::vector<float>& dout,
                               int M, int N, int d, std::vector<float>& dq,
                               std::vector<float>& dk, std::vector<float>& dv) {
    std::vector<double> gq(size_t(M) * d, 0), gk(size_t(N) * d, 0), gv(size_t(N) * d, 0);
    const double scale = 1.0 / std::sqrt(double(d));
    for (int r = 0; r < M; ++r) {
        std::vector<double> score(N, 0), dp(N, 0);
        for (int s = 0; s < N; ++s) {
            for (int j = 0; j < d; ++j) {
                score[s] += double(q[r * d + j]) * k[s * d + j];
                dp[s] += double(dout[r * d + j]) * v[s * d + j];
            }
            score[s] *= scale;
        }
        auto p = softmax(score);
        double dot = 0;
        for (int s = 0; s < N; ++s) dot += dp[s] * p[s];
        for (int s = 0; s < N; ++s) {
            const double ds = p[s] * (dp[s] - dot) * scale;
            for (int j = 0; j < d; ++j) {
                gv[s * d + j] += p[s] * dout[r * d + j];
                gq[r * d + j] += ds * k[s * d + j];
                gk[s * d + j] += ds * q[r * d + j];
            }
        }
    }
    dq.assign(gq.begin(), gq.end());
    dk.assign(gk.begin(), gk.end());
    dv.assign(gv.begin(), gv.end());
}
}  // namespace reference
