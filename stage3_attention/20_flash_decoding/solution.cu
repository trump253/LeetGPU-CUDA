#include "solution.h"

__global__ void flash_decoding_kernel(const float* Q, const float* K, const float* V, float* output, int T, int S,
                      int H, int D) {
    // TODO: implement your GPU kernel here.
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int T, int S,
                      int H, int D) {
    // TODO: choose a grid/block mapping for this operator.
    flash_decoding_kernel<<<1, 256>>>(Q, K, V, output, T, S, H, D);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}
