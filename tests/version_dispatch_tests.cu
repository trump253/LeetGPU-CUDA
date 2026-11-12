#include "bench.hpp"

// Test dispatch and state restoration without depending on any exercise answer.
__global__ void increment(float* state) { if (threadIdx.x == 0) *state += 1; }
extern "C" void solve(float* state);

namespace {
void one_kernel(float* state) { increment<<<1, 32>>>(state); }
void two_kernels(float* state) {
    increment<<<1, 32>>>(state);
    increment<<<1, 32>>>(state);
}
using Function = decltype(&solve);
constexpr solution::Version<Function> versions[] = {
    {0, "one_kernel", one_kernel}, {1, "two_kernels", two_kernels}
};
Function selected = one_kernel;
int selected_id = 0;
}

solution::VersionList list_solution_versions() { return solution::describe(versions, 0); }
void select_solution_version(int id) {
    selected = solution::find_launcher(versions, id);
    selected_id = id;
}
extern "C" void solve(float* state) { selected(state); }

int main(int argc, char** argv) {
    return bench::entry(argc, argv, "version_dispatch_tests", [](const bench::Options& options) {
        if (options.list) { std::cout << "0 [tiny] state reset\n"; return 0; }
        if (selected_id != options.selected_version)
            throw std::runtime_error("version was not selected before creating the fixture");
        bench::Fixture fixture(options, "tiny", "identical inout state; one vs two kernels", 0, 0);
        auto& state = fixture.inout<float>("state", 1);
        state.h[0] = 3;
        fixture.reference([&] { state.ref[0] = selected_id == 0 ? 4 : 5; });
        return fixture.run([&] { solve(state.data()); }) ? 0 : 1;
    }, 1, list_solution_versions, select_solution_version);
}
