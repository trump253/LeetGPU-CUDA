#include "reference.hpp"

#include <iostream>
#include <numeric>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void close(double actual, double expected, double tolerance = 1e-10) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
        throw std::runtime_error("reference mismatch: " + std::to_string(actual) + " vs " + std::to_string(expected));
}
double loss(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
            const std::vector<float>& dout, int M, int N, int D) {
    double result = 0;
    for (int r = 0; r < M; ++r) {
        auto y = reference::attention_row(N, D,
            [&](int j) { return q[r * D + j]; },
            [&](int s, int j) { return k[s * D + j]; },
            [&](int s, int j) { return v[s * D + j]; });
        for (int j = 0; j < D; ++j) result += y[j] * dout[r * D + j];
    }
    return result;
}
}

int main() {
    try {
        auto p = reference::softmax({1000, 1000, 1000});
        for (double x : p) close(x, 1. / 3);
        auto extreme = reference::softmax({-1000, 1000});
        close(extreme[0], 0); close(extreme[1], 1);
        auto a = reference::softmax({-1, 0, 1}), b = reference::softmax({999, 1000, 1001});
        for (size_t i = 0; i < a.size(); ++i) close(a[i], b[i]);
        close(std::accumulate(a.begin(), a.end(), 0.), 1.);
        bool rejected = false;
        try { reference::softmax({}); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "empty softmax must fail");

        // Uniform attention has a hand-computable result independent of softmax internals.
        auto uniform = reference::attention_row(3, 2,
            [](int) { return 0.; }, [](int s, int j) { return 2. * s + j; },
            [](int s, int j) { return 3. * s - j; });
        close(uniform[0], 3); close(uniform[1], 2);
        auto single = reference::attention_row(1, 3,
            [](int j) { return 100. * j; }, [](int, int j) { return 10. - j; },
            [](int, int j) { return -2. * j; });
        for (int j = 0; j < 3; ++j) close(single[j], -2. * j);

        // Check all three attention gradients against central finite differences.
        const int M = 2, N = 3, D = 2;
        std::vector<float> q{.2f, -.3f, .7f, .1f}, k{.1f, .6f, -.4f, .9f, .2f, -.5f};
        std::vector<float> v{1, -2, 3, .5f, -.7f, .4f}, dout{.3f, -.8f, .1f, .5f};
        std::vector<float> dq, dk, dv;
        reference::attention_backward(q, k, v, dout, M, N, D, dq, dk, dv);
        auto finite_difference = [&](std::vector<float>& x, const std::vector<float>& gradient) {
            for (size_t i = 0; i < x.size(); ++i) {
                float original = x[i];
                x[i] = original + .001f; double plus = loss(q, k, v, dout, M, N, D); float hi = x[i];
                x[i] = original - .001f; double minus = loss(q, k, v, dout, M, N, D); float lo = x[i];
                x[i] = original;
                close(gradient[i], (plus - minus) / double(hi - lo), 2e-6);
            }
        };
        finite_difference(q, dq); finite_difference(k, dk); finite_difference(v, dv);
        std::cout << "PASS: softmax stability, attention identities, dQ/dK/dV finite differences\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
