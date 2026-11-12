#include "solution.h"

__global__ void paged_attention_kernel(const float* Q, const float* K_cache, const float* V_cache,
                      const int* block_table, const int* context_lens, float* output,
                      int batch_size, int num_heads, int head_dim, int block_size,
                      int max_blocks_per_seq) {
    // TODO: implement your GPU kernel here.
}

extern "C" void solve(const float* Q, const float* K_cache, const float* V_cache,
                      const int* block_table, const int* context_lens, float* output,
                      int batch_size, int num_heads, int head_dim, int block_size,
                      int max_blocks_per_seq) {
    // TODO: choose a grid/block mapping for this operator.
    paged_attention_kernel<<<1, 256>>>(Q, K_cache, V_cache, block_table, context_lens, output, batch_size, num_heads, head_dim, block_size, max_blocks_per_seq);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}
