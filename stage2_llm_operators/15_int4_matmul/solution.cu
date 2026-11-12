#include "solution.h"

__global__ void int4_matmul_kernel(const __half* x, const uint8_t* w_q, const __half* scales, __half* y, int M,
                      int N, int K, int group_size) {
    // TODO: implement your GPU kernel here.
}

namespace {
// Each version owns its launch configuration and may launch multiple kernels.
void solve_v0(const __half* x, const uint8_t* w_q, const __half* scales, __half* y, int M,
                      int N, int K, int group_size) {
    // TODO: choose a grid/block mapping for this operator.
    int4_matmul_kernel<<<1, 256>>>(x, w_q, scales, y, M, N, K, group_size);
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

extern "C" void solve(const __half* x, const uint8_t* w_q, const __half* scales, __half* y, int M,
                      int N, int K, int group_size) {
    selected_impl(x, w_q, scales, y, M, N, K, group_size);
}
