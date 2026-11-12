#include "solution.h"

__global__ void grouped_query_attention_kernel(const float* Q, const float* K, const float* V, float* output,
                      int num_q_heads, int num_kv_heads, int seq_len, int head_dim) {
    // TODO: implement your GPU kernel here.
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int num_q_heads, int num_kv_heads, int seq_len, int head_dim) {
    // TODO: choose a grid/block mapping for this operator.
    grouped_query_attention_kernel<<<1, 256>>>(Q, K, V, output, num_q_heads, num_kv_heads, seq_len, head_dim);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}
