#include "solution.h"

__global__ void prefix_sum_kernel(const float* input, float* output, int N) {
    // TODO: implement your GPU kernel here.
}

extern "C" void solve(const float* input, float* output, int N) {
    // TODO: choose a grid/block mapping for this operator.
    prefix_sum_kernel<<<1, 256>>>(input, output, N);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}
