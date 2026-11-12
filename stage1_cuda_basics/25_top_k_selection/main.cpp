#include "solution.h"
#include "bench.hpp"
#include "reference.hpp"

using namespace bench;
namespace {
struct TestCase { const char* tier; int N; int k; const char* pattern; };
const TestCase cases[] = {
    {"tiny", 1, 1, "random"},
    {"small", 31, 1, "random"},
    {"small", 257, 257, "duplicates"},
    {"medium", 65539, 37, "random"},
    {"large", 4194307, 100, "duplicates"}
};
}

int main(int argc, char** argv) {
    return entry(argc, argv, "p25_top_k_selection", [&](const Options& options) {
        int failed=0, index=0;
        for(const auto& tc : cases) {
            int case_index=index++;
            if(!options.select(tc.tier,case_index)) continue;
            std::string label="N=" + std::to_string(tc.N) + ", " + "k=" + std::to_string(tc.k) + ", pattern=" + tc.pattern;
            if(options.list) { std::cout << case_index << " [" << tc.tier << "] " << label << '\n'; continue; }
            const int N=tc.N; const int k=tc.k;
            const std::string pattern=tc.pattern;
            std::mt19937 rng(options.seed + 25u*1009u + unsigned(case_index)*9176u);
            Fixture f(options,tc.tier,label,0,0);
            auto& x=f.input<float>("input",N); auto& o=f.output<float>("output",k); fill(x.h,rng,pattern);
            f.reference([&] {
                auto sorted=x.h; std::partial_sort(sorted.begin(),sorted.begin()+k,sorted.end(),std::greater<float>()); std::copy_n(sorted.begin(),k,o.ref.begin());
            });
            if(!f.run([&] { solve(x.data(), o.data(), N, k); })) ++failed;
        }
        return failed ? 1 : 0;
    }, std::size(cases));
}
