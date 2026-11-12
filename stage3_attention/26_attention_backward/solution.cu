#include "solution.h"

__global__ void attention_backward_kernel(const float* Q, const float* K, const float* V, const float* dO, float* dQ,
                      float* dK, float* dV, int M, int N, int d) {
    // TODO: implement your GPU kernel here.
}

extern "C" void solve(const float* Q, const float* K, const float* V, const float* dO, float* dQ,
                      float* dK, float* dV, int M, int N, int d) {
    // TODO: choose a grid/block mapping for this operator.
    attention_backward_kernel<<<1, 256>>>(Q, K, V, dO, dQ, dK, dV, M, N, d);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}
