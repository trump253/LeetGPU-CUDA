#include "solution.h"

__global__ void top_k_selection_kernel(const float* input, float* output, int N, int k) {
    // TODO: implement your GPU kernel here.
}

extern "C" void solve(const float* input, float* output, int N, int k) {
    // TODO: choose a grid/block mapping for this operator.
    top_k_selection_kernel<<<1, 256>>>(input, output, N, k);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}
