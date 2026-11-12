#include "solution.h"

__global__ void gemm_kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta) {
    // TODO: implement your GPU kernel here.
}

extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta) {
    // TODO: choose a grid/block mapping for this operator.
    gemm_kernel<<<1, 256>>>(A, B, C, M, N, K, alpha, beta);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}
