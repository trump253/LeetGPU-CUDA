#include "solution.h"
#include "bench.hpp"
#include "reference.hpp"

using namespace bench;
namespace {
struct TestCase { const char* tier; int N; int C; const char* pattern; };
const TestCase cases[] = {
    {"tiny", 1, 1, "random"},
    {"small", 3, 31, "constant"},
    {"small", 7, 257, "offset"},
    {"medium", 128, 1025, "random"},
    {"large", 1024, 4096, "random"}
};
}

int main(int argc, char** argv) {
    return entry(argc, argv, "p10_layer_normalization", [&](const Options& options) {
        int failed=0, index=0;
        for(const auto& tc : cases) {
            int case_index=index++;
            if(!options.select(tc.tier,case_index)) continue;
            std::string label="N=" + std::to_string(tc.N) + ", " + "C=" + std::to_string(tc.C) + ", pattern=" + tc.pattern;
            if(options.list) { std::cout << case_index << " [" << tc.tier << "] " << label << '\n'; continue; }
            const int N=tc.N; const int C=tc.C;
            const std::string pattern=tc.pattern;
            std::mt19937 rng(options.seed + 10u*1009u + unsigned(case_index)*9176u);
            Fixture f(options,tc.tier,label,3e-4,3e-4);
            auto& x=f.input<float>("input",size_t(N)*C); auto& w=f.input<float>("weight",C); auto& b=f.input<float>("bias",C); auto& o=f.output<float>("output",x.h.size());
            fill(x.h,rng,pattern); fill(w.h,rng,"random"); for(float& a:w.h) a=1.f+0.5f*a; fill(b.h,rng,"random"); float eps=1e-5f;
            f.reference([&] {
                for(int r=0;r<N;++r) {double mean=0,var=0; for(int c=0;c<C;++c) mean+=x.h[r*C+c]; mean/=C; for(int c=0;c<C;++c) {double z=x.h[r*C+c]-mean; var+=z*z;} for(int c=0;c<C;++c) o.ref[r*C+c]=float((x.h[r*C+c]-mean)/std::sqrt(var/C+eps)*w.h[c]+b.h[c]);}
            });
            if(!f.run([&] { solve(x.data(), w.data(), b.data(), o.data(), N, C, eps); })) ++failed;
        }
        return failed ? 1 : 0;
    }, std::size(cases), list_solution_versions, select_solution_version);
}
