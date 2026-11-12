#include "solution.h"

__global__ void int8_kv_attention_kernel(const float* Q, const int8_t* K_int8, const int8_t* V_int8,
                      const float* k_scale, const float* v_scale, float* output, int num_heads,
                      int seq_len, int head_dim) {
    // TODO: implement your GPU kernel here.
}

extern "C" void solve(const float* Q, const int8_t* K_int8, const int8_t* V_int8,
                      const float* k_scale, const float* v_scale, float* output, int num_heads,
                      int seq_len, int head_dim) {
    // TODO: choose a grid/block mapping for this operator.
    int8_kv_attention_kernel<<<1, 256>>>(Q, K_int8, V_int8, k_scale, v_scale, output, num_heads, seq_len, head_dim);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}
