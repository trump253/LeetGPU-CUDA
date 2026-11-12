#include "solution.h"

__global__ void fused_qkv_rope_cache_kernel(const float* x, const float* W_qkv, const float* cos_sin_cache,
                      const int* positions, float* K_cache, float* V_cache, float* Q_out, int B,
                      int d_model, int H_q, int H_kv, int D, int S_max) {
    // TODO: implement your GPU kernel here.
}

extern "C" void solve(const float* x, const float* W_qkv, const float* cos_sin_cache,
                      const int* positions, float* K_cache, float* V_cache, float* Q_out, int B,
                      int d_model, int H_q, int H_kv, int D, int S_max) {
    // TODO: choose a grid/block mapping for this operator.
    fused_qkv_rope_cache_kernel<<<1, 256>>>(x, W_qkv, cos_sin_cache, positions, K_cache, V_cache, Q_out, B, d_model, H_q, H_kv, D, S_max);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}
