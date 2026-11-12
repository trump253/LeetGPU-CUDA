#include "solution.h"

__global__ void split_kv_reduction_kernel(const float* partial_out, const float* partial_lse, float* output,
                      int num_splits, int num_heads, int head_dim) {
    // TODO: implement your GPU kernel here.
}

namespace {
// Each version owns its launch configuration and may launch multiple kernels.
void solve_v0(const float* partial_out, const float* partial_lse, float* output,
                      int num_splits, int num_heads, int head_dim) {
    // TODO: choose a grid/block mapping for this operator.
    split_kv_reduction_kernel<<<1, 256>>>(partial_out, partial_lse, output, num_splits, num_heads, head_dim);
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

extern "C" void solve(const float* partial_out, const float* partial_lse, float* output,
                      int num_splits, int num_heads, int head_dim) {
    selected_impl(partial_out, partial_lse, output, num_splits, num_heads, head_dim);
}
