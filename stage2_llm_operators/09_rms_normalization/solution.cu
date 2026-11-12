#include "solution.h"

__global__ void rms_normalization_kernel(const float* input, float gamma, float beta, float* output, int N,
                      float eps) {
    // TODO: implement your GPU kernel here.
}

extern "C" void solve(const float* input, float gamma, float beta, float* output, int N,
                      float eps) {
    // TODO: choose a grid/block mapping for this operator.
    rms_normalization_kernel<<<1, 256>>>(input, gamma, beta, output, N, eps);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}
