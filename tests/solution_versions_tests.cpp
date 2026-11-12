#include "solution_versions.hpp"

#include <iostream>

namespace {
int first(int value) { return value + 1; }
int second(int value) { return value + 2; }
using Function = int (*)(int);
constexpr solution::Version<Function> versions[] = {
    {0, "first", first}, {3, "second", second}
};
static_assert(solution::valid_versions(versions, 3), "valid nonconsecutive IDs rejected");
static_assert(!solution::valid_versions(versions, 1), "missing default accepted");
constexpr solution::Version<Function> duplicate_ids[] = {{0, "first", first}, {0, "second", second}};
constexpr solution::Version<Function> duplicate_names[] = {{0, "same", first}, {1, "same", second}};
constexpr solution::Version<Function> reserved_name[] = {{0, "all", first}};
constexpr solution::Version<Function> numeric_name[] = {{0, "3", first}};
constexpr solution::Version<Function> missing_launcher[] = {{0, "empty", nullptr}};
static_assert(!solution::valid_versions(duplicate_ids, 0), "duplicate IDs accepted");
static_assert(!solution::valid_versions(duplicate_names, 0), "duplicate names accepted");
static_assert(!solution::valid_versions(reserved_name, 0), "reserved selector accepted as name");
static_assert(!solution::valid_versions(numeric_name, 0), "numeric selector accepted as name");
static_assert(!solution::valid_versions(missing_launcher, 0), "null launcher accepted");

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <class Run> void rejects(Run run) {
    bool rejected = false;
    try { run(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "unknown/empty version was accepted");
}
}

int main() {
    try {
        const auto list = solution::describe(versions, 3);
        require(list.size() == 2 && !list[0].is_default && list[1].is_default, "incorrect explicit default");
        require(solution::find_launcher(versions, 0)(10) == 11, "incorrect first launcher");
        require(solution::find_launcher(versions, 3)(10) == 12, "incorrect second launcher");
        require(solution::resolve_versions(list, "default")[0].id == 3, "default selector failed");
        require(solution::resolve_versions(list, "0")[0].id == 0, "ID selector failed");
        require(solution::resolve_versions(list, "second")[0].id == 3, "name selector failed");
        auto all = solution::resolve_versions(list, "all");
        require(all.size() == 2 && all[0].id == 0 && all[1].id == 3, "all selector order failed");
        rejects([&] { solution::find_launcher(versions, 1); });
        rejects([&] { solution::resolve_versions(list, "missing"); });
        rejects([&] { solution::resolve_versions(list, "-1"); });
        rejects([&] { solution::resolve_versions({}, "all"); });
        std::cout << "PASS: version registration, explicit defaults, typed launchers and selectors\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
