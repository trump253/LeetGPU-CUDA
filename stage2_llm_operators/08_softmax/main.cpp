#include "solution.h"
#include "bench.hpp"
#include "reference.hpp"

using namespace bench;
namespace {
struct TestCase { const char* tier; int N; const char* pattern; };
const TestCase cases[] = {
    {"tiny", 1, "random"},
    {"small", 31, "zeros"},
    {"small", 257, "extreme"},
    {"small", 1003, "random"},
    {"medium", 65539, "random"},
    {"large", 500000, "extreme"}
};
}

int main(int argc, char** argv) {
    return entry(argc, argv, "p08_softmax", [&](const Options& options) {
        int failed=0, index=0;
        for(const auto& tc : cases) {
            int case_index=index++;
            if(!options.select(tc.tier,case_index)) continue;
            std::string label="N=" + std::to_string(tc.N) + ", pattern=" + tc.pattern;
            if(options.list) { std::cout << case_index << " [" << tc.tier << "] " << label << '\n'; continue; }
            const int N=tc.N;
            const std::string pattern=tc.pattern;
            std::mt19937 rng(options.seed + 8u*1009u + unsigned(case_index)*9176u);
            Fixture f(options,tc.tier,label,1e-7,2e-4);
            auto& x=f.input<float>("input",N); auto& o=f.output<float>("output",N); fill(x.h,rng,pattern);
            f.reference([&] {
                auto p=reference::softmax(std::vector<double>(x.h.begin(),x.h.end())); for(int j=0;j<N;++j) o.ref[j]=float(p[j]);
            });
            if(!f.run([&] { solve(x.data(), o.data(), N); })) ++failed;
        }
        return failed ? 1 : 0;
    }, std::size(cases));
}
