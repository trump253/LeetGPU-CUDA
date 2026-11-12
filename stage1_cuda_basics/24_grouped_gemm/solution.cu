#include "solution.h"

__global__ void grouped_gemm_kernel(const float* A, const float* B, const int* group_offsets, float* C, int G,
                      int M_total, int K, int N) {
    // TODO: implement your GPU kernel here.
}

namespace {
// Each version owns its launch configuration and may launch multiple kernels.
void solve_v0(const float* A, const float* B, const int* group_offsets, float* C, int G,
                      int M_total, int K, int N) {
    // TODO: choose a grid/block mapping for this operator.
    grouped_gemm_kernel<<<1, 256>>>(A, B, group_offsets, C, G, M_total, K, N);
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

extern "C" void solve(const float* A, const float* B, const int* group_offsets, float* C, int G,
                      int M_total, int K, int N) {
    selected_impl(A, B, group_offsets, C, G, M_total, K, N);
}
