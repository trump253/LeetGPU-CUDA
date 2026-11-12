#include "solution.h"

__global__ void dot_product_kernel(const float* A, const float* B, float* result, int N) {
    // TODO: implement your GPU kernel here.
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
    // TODO: choose a grid/block mapping for this operator.
    dot_product_kernel<<<1, 256>>>(A, B, result, N);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}
