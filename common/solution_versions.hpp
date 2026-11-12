#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace solution {
struct VersionInfo {
    int id;
    const char* name;
    bool is_default;
};
using VersionList = std::vector<VersionInfo>;

// Each entry points to a host launcher with the same signature as solve().
template <class Function>
struct Version {
    int id;
    const char* name;
    Function launch;
};

constexpr bool same_name(const char* a, const char* b) {
    if (!a || !b) return a == b;
    while (*a && *a == *b) { ++a; ++b; }
    return *a == *b;
}

template <class Function, std::size_t N>
constexpr bool valid_versions(const Version<Function> (&versions)[N], int default_id) {
    bool has_default = false;
    for (std::size_t i = 0; i < N; ++i) {
        const auto& v = versions[i];
        if (v.id < 0 || !v.name || !*v.name || !v.launch) return false;
        if ((*v.name >= '0' && *v.name <= '9') || same_name(v.name, "all") ||
            same_name(v.name, "default")) return false;
        for (std::size_t j = 0; j < i; ++j)
            if (v.id == versions[j].id || same_name(v.name, versions[j].name)) return false;
        has_default = has_default || v.id == default_id;
    }
    return has_default;
}

template <class Function, std::size_t N>
constexpr Function find_launcher(const Version<Function> (&versions)[N], int id) {
    for (const auto& v : versions) if (v.id == id) return v.launch;
    throw std::invalid_argument("Unknown solution version: " + std::to_string(id));
}

template <class Function, std::size_t N>
VersionList describe(const Version<Function> (&versions)[N], int default_id) {
    VersionList result;
    result.reserve(N);
    for (const auto& v : versions) result.push_back({v.id, v.name, v.id == default_id});
    return result;
}

// Resolve the CLI selector once, before CUDA initialization/warmup/timing.
inline VersionList resolve_versions(const VersionList& versions, const std::string& selector) {
    if (versions.empty()) throw std::invalid_argument("No solution versions registered");
    if (selector == "all") return versions;
    for (const auto& v : versions) {
        if ((selector == "default" && v.is_default) || selector == std::to_string(v.id) ||
            selector == v.name) return {v};
    }
    std::string available;
    for (const auto& v : versions) available += " " + std::to_string(v.id) + ":" + v.name;
    throw std::invalid_argument("Unknown solution version: " + selector + ". Available:" + available);
}
}  // namespace solution

// Every exercise provides these local testing hooks in its solution.cu.
// The public extern "C" solve() signature remains unchanged.
solution::VersionList list_solution_versions();
void select_solution_version(int version);
