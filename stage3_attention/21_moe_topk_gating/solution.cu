#include "solution.h"

__global__ void moe_topk_gating_kernel(const float* logits, float* topk_weights, int* topk_indices, int M, int E,
                      int k) {
    // TODO: implement your GPU kernel here.
}

extern "C" void solve(const float* logits, float* topk_weights, int* topk_indices, int M, int E,
                      int k) {
    // TODO: choose a grid/block mapping for this operator.
    moe_topk_gating_kernel<<<1, 256>>>(logits, topk_weights, topk_indices, M, E, k);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}
