#include "solution.h"

__global__ void swiglu_mlp_kernel(const float* x, const float* W_gate, const float* W_up, const float* W_down,
                      float* output, int M, int d_model, int d_ffn) {
    // TODO: implement your GPU kernel here.
}

extern "C" void solve(const float* x, const float* W_gate, const float* W_up, const float* W_down,
                      float* output, int M, int d_model, int d_ffn) {
    // TODO: choose a grid/block mapping for this operator.
    swiglu_mlp_kernel<<<1, 256>>>(x, W_gate, W_up, W_down, output, M, d_model, d_ffn);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}
