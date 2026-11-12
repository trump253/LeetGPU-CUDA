#include "solution.h"

__global__ void paged_attention_kernel(const float* Q, const float* K_cache, const float* V_cache,
                      const int* block_table, const int* context_lens, float* output,
                      int batch_size, int num_heads, int head_dim, int block_size,
                      int max_blocks_per_seq) {
    // TODO: implement your GPU kernel here.
}

namespace {
// Each version owns its launch configuration and may launch multiple kernels.
void solve_v0(const float* Q, const float* K_cache, const float* V_cache,
                      const int* block_table, const int* context_lens, float* output,
                      int batch_size, int num_heads, int head_dim, int block_size,
                      int max_blocks_per_seq) {
    // TODO: choose a grid/block mapping for this operator.
    paged_attention_kernel<<<1, 256>>>(Q, K_cache, V_cache, block_table, context_lens, output, batch_size, num_heads, head_dim, block_size, max_blocks_per_seq);
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

extern "C" void solve(const float* Q, const float* K_cache, const float* V_cache,
                      const int* block_table, const int* context_lens, float* output,
                      int batch_size, int num_heads, int head_dim, int block_size,
                      int max_blocks_per_seq) {
    selected_impl(Q, K_cache, V_cache, block_table, context_lens, output, batch_size, num_heads, head_dim, block_size, max_blocks_per_seq);
}
