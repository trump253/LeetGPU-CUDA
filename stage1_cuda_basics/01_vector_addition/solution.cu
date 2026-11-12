#include "solution.h"

__global__ void vector_addition_kernel(const float* A, const float* B, float* C, int N) {
    // TODO: implement your GPU kernel here.
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    // TODO: choose a grid/block mapping for this operator.
    vector_addition_kernel<<<1, 256>>>(A, B, C, N);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}
