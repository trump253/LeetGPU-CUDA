#include "solution.h"

__global__ void layer_normalization_kernel(const float* input, const float* weight, const float* bias, float* output,
                      int N, int C, float eps) {
    // TODO: implement your GPU kernel here.
}

extern "C" void solve(const float* input, const float* weight, const float* bias, float* output,
                      int N, int C, float eps) {
    // TODO: choose a grid/block mapping for this operator.
    layer_normalization_kernel<<<1, 256>>>(input, weight, bias, output, N, C, eps);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}
