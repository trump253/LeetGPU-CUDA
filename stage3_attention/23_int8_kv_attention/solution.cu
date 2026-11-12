#include "solution.h"

__global__ void int8_kv_attention_kernel(const float* Q, const int8_t* K_int8, const int8_t* V_int8,
                      const float* k_scale, const float* v_scale, float* output, int num_heads,
                      int seq_len, int head_dim) {
    // TODO: implement your GPU kernel here.
}

namespace {
// Each version owns its launch configuration and may launch multiple kernels.
void solve_v0(const float* Q, const int8_t* K_int8, const int8_t* V_int8,
                      const float* k_scale, const float* v_scale, float* output, int num_heads,
                      int seq_len, int head_dim) {
    // TODO: choose a grid/block mapping for this operator.
    int8_kv_attention_kernel<<<1, 256>>>(Q, K_int8, V_int8, k_scale, v_scale, output, num_heads, seq_len, head_dim);
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

extern "C" void solve(const float* Q, const int8_t* K_int8, const int8_t* V_int8,
                      const float* k_scale, const float* v_scale, float* output, int num_heads,
                      int seq_len, int head_dim) {
    selected_impl(Q, K_int8, V_int8, k_scale, v_scale, output, num_heads, seq_len, head_dim);
}
