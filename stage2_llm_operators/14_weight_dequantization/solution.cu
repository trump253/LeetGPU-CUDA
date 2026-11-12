#include "solution.h"

__global__ void weight_dequantization_kernel(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {
    // TODO: implement your GPU kernel here.
}

extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {
    // TODO: choose a grid/block mapping for this operator.
    weight_dequantization_kernel<<<1, 256>>>(X, S, Y, M, N, TILE_SIZE);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}
