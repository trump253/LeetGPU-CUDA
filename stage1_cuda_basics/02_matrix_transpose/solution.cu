#include "solution.h"

constexpr int TILE_SIZE = 32;
constexpr int BLOCK_THREADS_Y = 2;

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

__global__ void matrix_transpose_kernel_v1(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int rows, int cols) {
    int input_row = blockDim.y * blockIdx.y + threadIdx.y;
    int input_col = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float tile[16][17];
    if (input_row < rows && input_col < cols) {
        tile[threadIdx.x][threadIdx.y] = input[input_row * cols + input_col];
    }
    __syncthreads();
    int output_row = blockDim.x * blockIdx.x + threadIdx.y;
    int output_col = blockDim.y * blockIdx.y + threadIdx.x;
    if (output_row < cols && output_col < rows) {
        output[output_row * rows + output_col] = tile[threadIdx.y][threadIdx.x];
    }
}

__global__ void matrix_transpose_kernel_v2(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int rows, int cols) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    for (int offset = 0; offset < TILE_SIZE; offset += BLOCK_THREADS_Y) {
        int tile_row = row + offset;
        if (tile_row < rows && col < cols) {
            tile[threadIdx.y + offset][threadIdx.x] = input[tile_row * cols + col];
        }
    }

    __syncthreads();

    row = blockIdx.x * TILE_SIZE + threadIdx.y;
    col = blockIdx.y * TILE_SIZE + threadIdx.x;
    for (int offset = 0; offset < TILE_SIZE; offset += BLOCK_THREADS_Y) {
        int tile_row = row + offset;
        if (tile_row < cols && col < rows) {
            output[tile_row * rows + col] = tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

namespace {
// Each version owns its launch configuration and may launch multiple kernels.
void solve_v0(const float* input, float* output, int rows, int cols) {
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    matrix_transpose_kernel<<<grid, block>>>(input, output, rows, cols);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}

void solve_v1(const float* input, float* output, int rows, int cols) {
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    matrix_transpose_kernel_v1<<<grid, block>>>(input, output, rows, cols);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}

void solve_v2(const float* input, float* output, int rows, int cols) {
    dim3 block(TILE_SIZE, BLOCK_THREADS_Y);
    dim3 grid((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);
    matrix_transpose_kernel_v2<<<grid, block>>>(input, output, rows, cols);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}

using SolveFn = decltype(&solve);
// Add a solve_vN host launcher above, then register it here.
constexpr solution::Version<SolveFn> kVersions[] = {
    {0, "naive", solve_v0},
    {1, "tiled16_padding", solve_v1},
    {2, "tiled_multi_padding", solve_v2},
};
constexpr int kDefaultVersion = 2;
static_assert(solution::valid_versions(kVersions, kDefaultVersion),
              "Version IDs/names must be unique and the default must exist");
SolveFn selected_impl = solution::find_launcher(kVersions, kDefaultVersion);
}  // namespace

solution::VersionList list_solution_versions() {
    return solution::describe(kVersions, kDefaultVersion);
}

void select_solution_version(int version) {
    selected_impl = solution::find_launcher(kVersions, version);
}

extern "C" void solve(const float* input, float* output, int rows, int cols) {
    selected_impl(input, output, rows, cols);
}
