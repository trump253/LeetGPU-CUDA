#pragma once

#include "solution_versions.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace bench {
inline void check(cudaError_t error, const char* expression, const char* file, int line) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + " " +
                                 expression + ": " + cudaGetErrorString(error));
}
#define CUDA_CHECK(expr) ::bench::check((expr), #expr, __FILE__, __LINE__)

using Clock = std::chrono::steady_clock;
inline double milliseconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

struct Options {
    std::string size = "all";
    std::string version = "default";
    int selected_version = -1;
    std::string selected_version_name;
    int iterations = 10, warmup = 3, device = 0, case_index = -1;
    uint32_t seed = 42;
    bool self_test = false, check_only = false, list = false, list_versions = false, help = false;

    bool select(const char* tier, int index) const {
        return (size == "all" || size == tier) && (case_index < 0 || index == case_index);
    }
};

inline int integer(const std::string& value, const std::string& option, int minimum) {
    size_t used = 0;
    long long n;
    try { n = std::stoll(value, &used); }
    catch (...) { throw std::invalid_argument("Invalid integer for " + option + ": " + value); }
    if (used != value.size() || n < minimum || n > std::numeric_limits<int>::max())
        throw std::invalid_argument("Out-of-range integer for " + option + ": " + value);
    return int(n);
}

inline Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--self-test") o.self_test = true;
        else if (key == "--check-only") o.check_only = true;
        else if (key == "--list-cases") o.list = true;
        else if (key == "--list-versions") o.list_versions = true;
        else if (key == "--help" || key == "-h") o.help = true;
        else {
            if (key != "--size" && key != "--case" && key != "--iterations" && key != "--warmup" &&
                key != "--device" && key != "--seed" && key != "--version")
                throw std::invalid_argument("Unknown option: " + key);
            if (++i == argc) throw std::invalid_argument("Missing value for " + key);
            std::string arg = argv[i];
            if (key == "--size") o.size = arg;
            else if (key == "--version") o.version = arg;
            else if (key == "--case") o.case_index = integer(arg, key, 0);
            else if (key == "--iterations") o.iterations = integer(arg, key, 1);
            else if (key == "--warmup") o.warmup = integer(arg, key, 0);
            else if (key == "--device") o.device = integer(arg, key, 0);
            else if (key == "--seed") o.seed = uint32_t(integer(arg, key, 0));
        }
    }
    if (o.size != "all" && o.size != "tiny" && o.size != "small" && o.size != "medium" && o.size != "large")
        throw std::invalid_argument("--size must be all/tiny/small/medium/large");
    if (o.case_index >= 0 && o.size != "all")
        throw std::invalid_argument("Use --case or --size, not both");
    return o;
}

template <class Run>
int entry(int argc, char** argv, const char* name, Run run, size_t cases,
          solution::VersionList (*available_versions)(), void (*select_version)(int)) {
    try {
        Options options = parse(argc, argv);
        if (options.help) {
            std::cout << name << "\n"
                "  --size all|tiny|small|medium|large  (default all)\n"
                "  --case INDEX        run one case from --list-cases\n"
                "  --version ID|NAME|all|default  (default: exercise's explicit default)\n"
                "  --list-versions     list registered versions without requiring a GPU\n"
                "  --iterations N      measured repetitions (default 10)\n"
                "  --warmup N          warmup repetitions (default 3)\n"
                "  --seed N            reproducible seed (default 42)\n"
                "  --device N          CUDA device ordinal (default 0)\n"
                "  --check-only        validate without timing\n"
                "  --list-cases        list cases without requiring a GPU\n"
                "  --self-test         test data transfer/checker; does NOT call solve\n"
                "Exit: 0=pass, 1=wrong result, 2=argument/CUDA/runtime error\n";
            return 0;
        }
        if (options.case_index >= int(cases)) throw std::invalid_argument("--case index out of range");
        const auto versions = available_versions();
        if (options.list_versions) {
            std::cout << name << " versions:\n";
            for (const auto& v : versions)
                std::cout << "  " << v.id << "  " << v.name << (v.is_default ? " (default)" : "") << '\n';
            return 0;
        }
        const auto selected = solution::resolve_versions(versions, options.version);
        if (options.list) return run(options);
        {
            CUDA_CHECK(cudaSetDevice(options.device));
            CUDA_CHECK(cudaFree(nullptr));  // initialize context outside timing
            cudaDeviceProp prop{};
            CUDA_CHECK(cudaGetDeviceProperties(&prop, options.device));
            std::cout << name << " | GPU " << options.device << ": " << prop.name
                      << " | seed=" << options.seed << " | warmup=" << options.warmup
                      << " | iterations=" << options.iterations << '\n';
            if (options.self_test)
                std::cout << "SELFTEST: CPU reference -> device -> host only; no solution/performance claim.\n";
        }
        int result = 0;
        for (const auto& v : selected) {
            // All selection and string handling happen outside Fixture::run's timed calls.
            select_version(v.id);
            options.selected_version = v.id;
            options.selected_version_name = v.name;
            std::cout << "=== version " << v.id << ": " << v.name
                      << (v.is_default ? " (default)" : "") << " ===\n";
            result = std::max(result, run(options));
        }
        return result;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 2;
    }
}

template <class T> inline double value(T v) { return double(v); }
template <> inline double value(half v) { return double(__half2float(v)); }
template <class T> inline T number(float v) { return T(v); }
template <> inline half number(float v) { return __float2half(v); }

template <class T>
void fill(std::vector<T>& values, std::mt19937& rng, const std::string& pattern, float scale = 1) {
    // Discrete binary fractions keep long scans/reductions reproducible and cancellation controlled.
    for (size_t i = 0; i < values.size(); ++i) {
        float v = float(int(rng() % 2049) - 1024) / 1024.f;
        if (pattern == "zeros") v = 0;
        else if (pattern == "constant") v = 3;
        else if (pattern == "signed") v = (i % 2 ? -1.f : 1.f) * float(i % 7 + 1) / 8;
        else if (pattern == "offset") v = 99 + v / 16;
        else if (pattern == "extreme") v *= 80;
        else if (pattern == "duplicates") v = float(int(rng() % 17) - 8);
        values[i] = number<T>(v * scale);
    }
}

enum class Role { Input, Output, InOut };
struct BufferBase {
    virtual ~BufferBase() = default;
    virtual void upload() = 0;
    virtual void reset_output() = 0;
    virtual void upload_reference() = 0;
    virtual bool verify(double atol, double rtol, bool verbose) = 0;
    virtual bool finite_reference() const = 0;
    virtual size_t device_bytes() const = 0;
};

template <class T>
class Buffer : public BufferBase {
    static constexpr size_t guard_size = 256;
    static constexpr unsigned char canary = 0xA5;
    unsigned char* allocation_ = nullptr;
    T* pointer_ = nullptr;
    Role role_;
    std::string name_;
    bool guards_ok(bool verbose) {
        unsigned char before[guard_size], after[guard_size];
        CUDA_CHECK(cudaMemcpy(before, allocation_, guard_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(after, allocation_ + guard_size + h.size() * sizeof(T), guard_size,
                              cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < guard_size; ++i)
            if (before[i] != canary || after[i] != canary) {
                if (verbose) std::cerr << "  " << name_ << ": out-of-bounds write in guard region\n";
                return false;
            }
        return true;
    }

public:
    std::vector<T> h, ref;
    Buffer(std::string name, size_t count, Role role)
        : role_(role), name_(std::move(name)), h(count), ref(role == Role::Input ? 0 : count) {
        if (!count) throw std::invalid_argument("Empty buffer: " + name_);
        if (role != Role::Input) {
            T poison;
            if constexpr (std::is_integral_v<T>) poison = std::numeric_limits<T>::max();
            else poison = number<T>(std::numeric_limits<float>::quiet_NaN());
            std::fill(ref.begin(), ref.end(), poison);
            if (role == Role::Output) std::fill(h.begin(), h.end(), poison);
        }
    }
    ~Buffer() override { if (allocation_) cudaFree(allocation_); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T* data() { return pointer_; }
    size_t device_bytes() const override { return h.size() * sizeof(T) + 2 * guard_size; }
    void upload() override {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&allocation_), device_bytes()));
        pointer_ = reinterpret_cast<T*>(allocation_ + guard_size);
        CUDA_CHECK(cudaMemset(allocation_, canary, guard_size));
        CUDA_CHECK(cudaMemset(allocation_ + guard_size + h.size() * sizeof(T), canary, guard_size));
        CUDA_CHECK(cudaMemcpy(pointer_, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    void reset_output() override {
        if (role_ != Role::Input)
            CUDA_CHECK(cudaMemcpy(pointer_, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    void upload_reference() override {
        if (role_ != Role::Input)
            CUDA_CHECK(cudaMemcpy(pointer_, ref.data(), ref.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    bool finite_reference() const override {
        for (auto v : ref) if (!std::isfinite(value(v))) return false;
        return true;
    }
    bool verify(double atol, double rtol, bool verbose) override {
        bool ok = guards_ok(verbose);
        std::vector<T> got(h.size());
        CUDA_CHECK(cudaMemcpy(got.data(), pointer_, got.size() * sizeof(T), cudaMemcpyDeviceToHost));
        if (role_ == Role::Input) {
            if (std::memcmp(h.data(), got.data(), got.size() * sizeof(T)) != 0) {
                if (verbose) std::cerr << "  " << name_ << ": read-only input modified\n";
                return false;
            }
            return ok;
        }
        size_t bad = 0, first = 0;
        double max_error = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            double actual = value(got[i]), expected = value(ref[i]);
            double error = std::abs(actual - expected);
            bool match;
            if constexpr (std::is_integral_v<T>) match = got[i] == ref[i];
            else match = std::isfinite(actual) && std::isfinite(expected) && error <= atol + rtol * std::abs(expected);
            if (!match) { if (!bad) first = i; ++bad; }
            max_error = std::max(max_error, std::isfinite(error) ? error : std::numeric_limits<double>::infinity());
        }
        if (bad && verbose)
            std::cerr << "  " << name_ << ": " << bad << "/" << got.size() << " mismatches, first["
                      << first << "] actual=" << value(got[first]) << " expected=" << value(ref[first])
                      << " max_abs=" << max_error << '\n';
        return ok && !bad;
    }
};

class Event {
    cudaEvent_t event_{};
public:
    Event() { CUDA_CHECK(cudaEventCreate(&event_)); }
    ~Event() { cudaEventDestroy(event_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    operator cudaEvent_t() const { return event_; }
};

struct Statistics {
    double mean, minimum, median, p95;
};
inline Statistics stats(std::vector<double> values) {
    if (values.empty()) throw std::invalid_argument("No timing samples");
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    return {std::accumulate(values.begin(), values.end(), 0.) / n, values.front(),
            n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2,
            values[size_t(std::ceil(0.95 * n)) - 1]};
}

class Fixture {
    const Options& options_;
    std::string tier_, label_;
    double atol_, rtol_, reference_ms_ = 0;
    std::vector<std::unique_ptr<BufferBase>> buffers_;

    template <class T> Buffer<T>& create(const char* name, size_t count, Role role) {
        auto buffer = std::make_unique<Buffer<T>>(name, count, role);
        auto& result = *buffer;
        buffers_.push_back(std::move(buffer));
        return result;
    }
    void reset() { for (auto& b : buffers_) b->reset_output(); }
    bool verify() {
        bool ok = true;
        for (auto& b : buffers_) if (!b->verify(atol_, rtol_, true)) ok = false;
        return ok;
    }

public:
    Fixture(const Options& options, std::string tier, std::string label, double atol, double rtol)
        : options_(options), tier_(std::move(tier)), label_(std::move(label)), atol_(atol), rtol_(rtol) {}
    template <class T> Buffer<T>& input(const char* name, size_t count) { return create<T>(name, count, Role::Input); }
    template <class T> Buffer<T>& output(const char* name, size_t count) { return create<T>(name, count, Role::Output); }
    template <class T> Buffer<T>& inout(const char* name, size_t count) { return create<T>(name, count, Role::InOut); }
    template <class Reference> void reference(Reference compute) {
        auto start = Clock::now();
        compute();
        reference_ms_ = milliseconds(start, Clock::now());
    }

    template <class Solve> bool run(Solve solve) {
        size_t bytes = 0;
        for (auto& b : buffers_) {
            if (!b->finite_reference()) throw std::runtime_error("CPU reference has NaN/Inf or unwritten values: " + label_);
            bytes += b->device_bytes();
        }
        if (options_.selected_version >= 0)
            std::cout << "[version=" << options_.selected_version << ":"
                      << options_.selected_version_name << "] ";
        std::cout << "[" << tier_ << "] " << label_ << " | device_MiB="
                  << std::fixed << std::setprecision(2) << bytes / 1048576.0
                  << " | cpu_ref_ms=" << reference_ms_ << std::endl;
        for (auto& b : buffers_) b->upload();
        if (options_.self_test) {
            for (auto& b : buffers_) b->upload_reference();
            bool ok = verify();
            std::cout << "  " << (ok ? "SELFTEST PASS" : "SELFTEST FAIL") << '\n';
            return ok;
        }
        for (int pass = 0; pass < 2; ++pass) {
            reset();
            solve();
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            if (!verify()) {
                std::cout << "  FAIL | timing skipped (implement/check solution.cu)\n";
                return false;
            }
        }
        if (options_.check_only) { std::cout << "  PASS | check only\n"; return true; }
        for (int i = 0; i < options_.warmup; ++i) {
            reset();
            solve();
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        Event start, stop;
        std::vector<double> gpu_us, wall_us;
        for (int i = 0; i < options_.iterations; ++i) {
            reset();  // restore C_initial / caches and poison outputs OUTSIDE event interval
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(start));
            auto wall_start = Clock::now();
            solve();
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            auto wall_stop = Clock::now();
            float elapsed = 0;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
            gpu_us.push_back(double(elapsed) * 1000);
            wall_us.push_back(milliseconds(wall_start, wall_stop) * 1000);
        }
        // Covers timing-loop output/state corruption as well as the initial correctness passes.
        if (!verify()) { std::cout << "  FAIL | post-benchmark check\n"; return false; }
        auto gpu = stats(gpu_us), wall = stats(wall_us);
        std::cout << std::fixed << std::setprecision(3)
                  << "  PASS | gpu_us mean=" << gpu.mean << " min=" << gpu.minimum
                  << " median=" << gpu.median << " p95=" << gpu.p95
                  << " | solve_wall_us mean=" << wall.mean << '\n';
        return true;
    }
};
}  // namespace bench
