#include "solution.h"

__global__ void fused_residual_rms_norm_kernel(const float* x, const float* residual, const float* weight, float* out, int N,
                      int C, float eps) {
    // TODO: implement your GPU kernel here.
}

extern "C" void solve(const float* x, const float* residual, const float* weight, float* out, int N,
                      int C, float eps) {
    // TODO: choose a grid/block mapping for this operator.
    fused_residual_rms_norm_kernel<<<1, 256>>>(x, residual, weight, out, N, C, eps);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}
