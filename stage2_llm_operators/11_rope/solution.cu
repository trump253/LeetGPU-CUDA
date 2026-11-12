#include "solution.h"

__global__ void rope_kernel(float* Q, float* cos, float* sin, float* output, int M, int D) {
    // TODO: implement your GPU kernel here.
}

extern "C" void solve(float* Q, float* cos, float* sin, float* output, int M, int D) {
    // TODO: choose a grid/block mapping for this operator.
    rope_kernel<<<1, 256>>>(Q, cos, sin, output, M, D);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}
