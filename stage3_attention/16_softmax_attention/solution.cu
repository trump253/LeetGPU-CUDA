#include "solution.h"

__global__ void softmax_attention_kernel(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d) {
    // TODO: implement your GPU kernel here.
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d) {
    // TODO: choose a grid/block mapping for this operator.
    softmax_attention_kernel<<<1, 256>>>(Q, K, V, output, M, N, d);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}
