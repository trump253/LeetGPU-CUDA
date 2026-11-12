#include "solution.h"

__global__ void split_kv_reduction_kernel(const float* partial_out, const float* partial_lse, float* output,
                      int num_splits, int num_heads, int head_dim) {
    // TODO: implement your GPU kernel here.
}

extern "C" void solve(const float* partial_out, const float* partial_lse, float* output,
                      int num_splits, int num_heads, int head_dim) {
    // TODO: choose a grid/block mapping for this operator.
    split_kv_reduction_kernel<<<1, 256>>>(partial_out, partial_lse, output, num_splits, num_heads, head_dim);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}
