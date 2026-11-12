#include "solution.h"
#include "bench.hpp"
#include "reference.hpp"

using namespace bench;
namespace {
struct TestCase { const char* tier; int N; const char* pattern; };
const TestCase cases[] = {
    {"tiny", 1, "random"},
    {"small", 31, "zeros"},
    {"small", 257, "constant"},
    {"medium", 8193, "random"},
    {"large", 100000, "random"}
};
}

int main(int argc, char** argv) {
    return entry(argc, argv, "p09_rms_normalization", [&](const Options& options) {
        int failed=0, index=0;
        for(const auto& tc : cases) {
            int case_index=index++;
            if(!options.select(tc.tier,case_index)) continue;
            std::string label="N=" + std::to_string(tc.N) + ", pattern=" + tc.pattern;
            if(options.list) { std::cout << case_index << " [" << tc.tier << "] " << label << '\n'; continue; }
            const int N=tc.N;
            const std::string pattern=tc.pattern;
            std::mt19937 rng(options.seed + 9u*1009u + unsigned(case_index)*9176u);
            Fixture f(options,tc.tier,label,1e-4,1e-4);
            auto& x=f.input<float>("input",N); auto& o=f.output<float>("output",N); fill(x.h,rng,pattern);
            float gamma=1.7f,beta=-0.3f,eps=1e-5f;
            f.reference([&] {
                double ss=0; for(float v:x.h) ss+=double(v)*v; for(int j=0;j<N;++j) o.ref[j]=float(gamma*x.h[j]/std::sqrt(ss/N+eps)+beta);
            });
            if(!f.run([&] { solve(x.data(), gamma, beta, o.data(), N, eps); })) ++failed;
        }
        return failed ? 1 : 0;
    }, std::size(cases), list_solution_versions, select_solution_version);
}
