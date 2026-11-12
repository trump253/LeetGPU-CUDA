#include "solution.h"

__global__ void top_k_selection_kernel(const float* input, float* output, int N, int k) {
    // TODO: implement your GPU kernel here.
}

namespace {
// Each version owns its launch configuration and may launch multiple kernels.
void solve_v0(const float* input, float* output, int N, int k) {
    // TODO: choose a grid/block mapping for this operator.
    top_k_selection_kernel<<<1, 256>>>(input, output, N, k);
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

extern "C" void solve(const float* input, float* output, int N, int k) {
    selected_impl(input, output, N, k);
}
