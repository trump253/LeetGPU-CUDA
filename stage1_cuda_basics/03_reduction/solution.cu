#include "solution.h"

constexpr int WARP_SIZE = 32;
constexpr unsigned FULL_MASK = 0xffffffffu;

constexpr int BLOCK_SIZE = 1024;
constexpr int WARP_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

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

namespace {
// Each version owns its launch configuration and may launch multiple kernels.
void solve_v0(const float* input, float* output, int N) {
    // TODO: choose a grid/block mapping for this operator.
    reduction_kernel<<<1, 1024>>>(input, output, N);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}

using SolveFn = decltype(&solve);
// Add a solve_vN host launcher above, then register it here.
constexpr solution::Version<SolveFn> kVersions[] = {
    {0, "starter", solve_v0},
};
constexpr int kDefaultVersion = 0;
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
