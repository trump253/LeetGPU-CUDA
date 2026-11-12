#include "solution.h"

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    // TODO: implement your GPU kernel here.
}

extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    matrix_transpose_kernel<<<grid, block>>>(input, output, rows, cols);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}
