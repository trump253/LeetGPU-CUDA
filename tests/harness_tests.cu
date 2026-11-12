#include "bench.hpp"

// Only tests harness state restoration/timing; this is not an exercise solution.
__global__ void increment_scalar(float* value) { if (threadIdx.x == 0) *value += 1; }

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main() {
    using namespace bench;
    try {
        CUDA_CHECK(cudaSetDevice(0));
        Buffer<float> output("output", 3, Role::Output);
        output.ref = {1, 0, -1};
        output.upload();
        require(!output.verify(1e-4, 1e-4, false), "unwritten NaN output was accepted");
        output.upload_reference();
        require(output.verify(0, 0, false), "exact output was rejected");
        float bad = std::numeric_limits<float>::infinity();
        CUDA_CHECK(cudaMemcpy(output.data(), &bad, sizeof(bad), cudaMemcpyHostToDevice));
        require(!output.verify(1, 1, false), "infinite output was accepted");
        output.upload_reference();
        unsigned char corrupt = 0;
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<unsigned char*>(output.data()) + 3 * sizeof(float),
                              &corrupt, 1, cudaMemcpyHostToDevice));
        require(!output.verify(0, 0, false), "guard corruption was accepted");

        Buffer<float> input("read_only", 1, Role::Input);
        input.h = {1}; input.upload();
        float changed = 2;
        CUDA_CHECK(cudaMemcpy(input.data(), &changed, sizeof(changed), cudaMemcpyHostToDevice));
        require(!input.verify(1, 1, false), "modified input was accepted");

        Buffer<int> integer_output("integers", 1, Role::Output);
        integer_output.ref[0] = 0; integer_output.upload();
        int one = 1;
        CUDA_CHECK(cudaMemcpy(integer_output.data(), &one, sizeof(one), cudaMemcpyHostToDevice));
        require(!integer_output.verify(100, 100, false), "integer tolerance must be exact");

        Options options; options.iterations = 4; options.warmup = 2;
        Fixture fixture(options, "test", "state reset and event timing", 0, 0);
        auto& state = fixture.inout<float>("state", 1);
        state.h[0] = 3;
        fixture.reference([&] { state.ref[0] = 4; });
        int calls = 0;
        require(fixture.run([&] { ++calls; increment_scalar<<<1, 32>>>(state.data()); }), "state not reset before every call");
        require(calls == 2 + options.warmup + options.iterations, "incorrect number of solve calls");
        auto s = stats({4, 1, 3, 2});
        require(s.mean == 2.5 && s.median == 2.5 && s.minimum == 1 && s.p95 == 4, "bad statistics");
        bool rejected = false;
        try { integer("1x", "--iterations", 1); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "malformed CLI integer accepted");
        std::cout << "PASS: poison, nonfinite, guard, readonly, integer, reset, timing and CLI checks\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
