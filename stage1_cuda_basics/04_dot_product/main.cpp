#include "solution.h"
#include "bench.hpp"
#include "reference.hpp"

using namespace bench;
namespace {
struct TestCase { const char* tier; int N; const char* pattern; };
const TestCase cases[] = {
    {"tiny", 1, "random"},
    {"small", 31, "signed"},
    {"small", 257, "zeros"},
    {"small", 1003, "random"},
    {"medium", 65539, "random"},
    {"large", 4194307, "random"}
};
}

int main(int argc, char** argv) {
    return entry(argc, argv, "p04_dot_product", [&](const Options& options) {
        int failed=0, index=0;
        for(const auto& tc : cases) {
            int case_index=index++;
            if(!options.select(tc.tier,case_index)) continue;
            std::string label="N=" + std::to_string(tc.N) + ", pattern=" + tc.pattern;
            if(options.list) { std::cout << case_index << " [" << tc.tier << "] " << label << '\n'; continue; }
            const int N=tc.N;
            const std::string pattern=tc.pattern;
            std::mt19937 rng(options.seed + 4u*1009u + unsigned(case_index)*9176u);
            Fixture f(options,tc.tier,label,2e-3,2e-4);
            auto& a=f.input<float>("A",N); auto& b=f.input<float>("B",N); auto& o=f.output<float>("result",1); fill(a.h,rng,pattern); fill(b.h,rng,pattern);
            f.reference([&] {
                double sum=0; for(int j=0;j<N;++j) sum+=double(a.h[j])*b.h[j]; o.ref[0]=float(sum);
            });
            if(!f.run([&] { solve(a.data(), b.data(), o.data(), N); })) ++failed;
        }
        return failed ? 1 : 0;
    }, std::size(cases), list_solution_versions, select_solution_version);
}
