#include "solution.h"

constexpr int WARP_SIZE = 32;
constexpr unsigned FULL_MASK = 0xffffffffu;

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
constexpr int ITEMS_PER_THREAD = 4;
constexpr int ITEMS_PER_BLOCK = ITEMS_PER_THREAD * BLOCK_SIZE;

__inline__ __device__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int delta = WARP_SIZE >> 1; delta > 0; delta >>= 1) {
        v += __shfl_down_sync(FULL_MASK, v, delta);
    }
    return v;
}

__global__ void reduction_kernel(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp = tid / WARP_SIZE;

    float thread_sum = 0.0f;
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        thread_sum += input[i];
    }

    __shared__ float shared_sum[WARP_PER_BLOCK];

    float warp_sum = warp_reduce_sum(thread_sum);
    if (lane == 0) {
        shared_sum[warp] = warp_sum;
    }
    __syncthreads();

    if (warp == 0) {
        float val = lane < WARP_PER_BLOCK ? shared_sum[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) {
            output[0] = val;
        }
    }
}

__global__ void reduction_kernel_v1(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp = tid / WARP_SIZE;
    int start = blockIdx.x * blockDim.x + tid;
    int step = blockDim.x * gridDim.x;

    float thread_sum = 0.0f;
    for (int i = start; i < N; i += step) {
        thread_sum += input[i];
    }

    __shared__ float shared_sum[WARP_PER_BLOCK];
    float warp_sum = warp_reduce_sum(thread_sum);
    if (lane == 0) {
        shared_sum[warp] = warp_sum;
    }
    __syncthreads();

    if (warp == 0) {
        float val = lane < WARP_PER_BLOCK ? shared_sum[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) {
            output[blockIdx.x] = val;
        }
    }
}

namespace {
// Each version owns its launch configuration and may launch multiple kernels.
void solve_v0(const float* input, float* output, int N) {
    reduction_kernel<<<1, BLOCK_SIZE>>>(input, output, N);
}

void solve_v1(const float* input, float* output, int N) {
    int G = (N + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK;
    if (G == 1) {
        reduction_kernel_v1<<<1, BLOCK_SIZE>>>(input, output, N);
        return;
    }
    float* partial = nullptr;
    cudaMalloc(&partial, G * sizeof(float));

    reduction_kernel_v1<<<G, BLOCK_SIZE>>>(input, partial, N);
    reduction_kernel_v1<<<1, BLOCK_SIZE>>>(partial, output, G);

    cudaFree(partial);
}

using SolveFn = decltype(&solve);
// Add a solve_vN host launcher above, then register it here.
constexpr solution::Version<SolveFn> kVersions[] = {
    {0, "starter", solve_v0},
    {1, "multi_block", solve_v1},
};
constexpr int kDefaultVersion = 1;
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

extern "C" void solve(const float* input, float* output, int N) {
    selected_impl(input, output, N);
}
