#include "solution.h"

__global__ void int4_matmul_kernel(const __half* x, const uint8_t* w_q, const __half* scales, __half* y, int M,
                      int N, int K, int group_size) {
    // TODO: implement your GPU kernel here.
}

extern "C" void solve(const __half* x, const uint8_t* w_q, const __half* scales, __half* y, int M,
                      int N, int K, int group_size) {
    // TODO: choose a grid/block mapping for this operator.
    int4_matmul_kernel<<<1, 256>>>(x, w_q, scales, y, M, N, K, group_size);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}
