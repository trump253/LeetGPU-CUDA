#include "solution.h"

__global__ void grouped_gemm_kernel(const float* A, const float* B, const int* group_offsets, float* C, int G,
                      int M_total, int K, int N) {
    // TODO: implement your GPU kernel here.
}

extern "C" void solve(const float* A, const float* B, const int* group_offsets, float* C, int G,
                      int M_total, int K, int N) {
    // TODO: choose a grid/block mapping for this operator.
    grouped_gemm_kernel<<<1, 256>>>(A, B, group_offsets, C, G, M_total, K, N);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}
