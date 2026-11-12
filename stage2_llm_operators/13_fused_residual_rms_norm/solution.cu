#include "solution.h"

__global__ void fused_residual_rms_norm_kernel(const float* x, const float* residual, const float* weight, float* out, int N,
                      int C, float eps) {
    // TODO: implement your GPU kernel here.
}

namespace {
// Each version owns its launch configuration and may launch multiple kernels.
void solve_v0(const float* x, const float* residual, const float* weight, float* out, int N,
                      int C, float eps) {
    // TODO: choose a grid/block mapping for this operator.
    fused_residual_rms_norm_kernel<<<1, 256>>>(x, residual, weight, out, N, C, eps);
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

extern "C" void solve(const float* x, const float* residual, const float* weight, float* out, int N,
                      int C, float eps) {
    selected_impl(x, residual, weight, out, N, C, eps);
}
