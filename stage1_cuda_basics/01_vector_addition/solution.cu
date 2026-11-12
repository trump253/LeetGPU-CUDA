#include "solution.h"

__global__ void vector_addition4_kernel(const float4* __restrict__ A,
                                        const float4* __restrict__ B,
                                        float4* __restrict__ C, int N4) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N4) {
        float4 a = A[idx];
        float4 b = B[idx];
        C[idx] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
}

__global__ void vector_addition_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C, int start, int N) {
    int idx = start + blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

namespace {
// Each version owns its launch configuration and may launch multiple kernels.
void solve_v0(const float* A, const float* B, float* C, int N) {
    if (N > 0) {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        vector_addition_kernel<<<blocks, threads>>>(A, B, C, 0, N);
    }
}

void solve_v1(const float* A, const float* B, float* C, int N) {
    int N4 = N >> 2;
    int tail = N & 3;
    if (N4 > 0) {
        const float4* A4 = reinterpret_cast<const float4*>(A);
        const float4* B4 = reinterpret_cast<const float4*>(B);
        float4* C4 = reinterpret_cast<float4*>(C);

        int threads = 256;
        int blocks = (N4 + threads - 1) / threads;
        vector_addition4_kernel<<<blocks, threads>>>(A4, B4, C4, N4);
    }
    if (tail > 0) {
        vector_addition_kernel<<<1, 32>>>(A, B, C, N & ~3, N);
    }
}

using SolveFn = decltype(&solve);
// Add a solve_vN host launcher above, then register it here.
constexpr solution::Version<SolveFn> kVersions[] = {
    {0, "scalar", solve_v0},
    {1, "float4_tail", solve_v1},
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

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    selected_impl(A, B, C, N);
}
