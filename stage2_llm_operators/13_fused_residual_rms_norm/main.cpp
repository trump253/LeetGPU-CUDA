#include "solution.h"
#include "bench.hpp"
#include "reference.hpp"

using namespace bench;
namespace {
struct TestCase { const char* tier; int N; int C; const char* pattern; };
const TestCase cases[] = {
    {"tiny", 1, 1, "random"},
    {"small", 3, 31, "cancel"},
    {"small", 7, 257, "random"},
    {"medium", 128, 1025, "random"},
    {"large", 1024, 4096, "random"}
};
}

int main(int argc, char** argv) {
    return entry(argc, argv, "p13_fused_residual_rms_norm", [&](const Options& options) {
        int failed=0, index=0;
        for(const auto& tc : cases) {
            int case_index=index++;
            if(!options.select(tc.tier,case_index)) continue;
            std::string label="N=" + std::to_string(tc.N) + ", " + "C=" + std::to_string(tc.C) + ", pattern=" + tc.pattern;
            if(options.list) { std::cout << case_index << " [" << tc.tier << "] " << label << '\n'; continue; }
            const int N=tc.N; const int C=tc.C;
            const std::string pattern=tc.pattern;
            std::mt19937 rng(options.seed + 13u*1009u + unsigned(case_index)*9176u);
            Fixture f(options,tc.tier,label,1e-4,1e-4);
            auto& x=f.input<float>("x",size_t(N)*C); auto& r=f.input<float>("residual",x.h.size()); auto& w=f.input<float>("weight",C); auto& o=f.output<float>("out",x.h.size()); fill(x.h,rng,"random"); fill(r.h,rng,"random"); fill(w.h,rng,"random"); for(float& a:w.h) a=1.f+0.5f*a;
            if(pattern=="cancel") for(size_t j=0;j<x.h.size();++j) r.h[j]=-x.h[j]; float eps=1e-5f;
            f.reference([&] {
                for(int row=0;row<N;++row) {double ss=0; for(int c=0;c<C;++c) {float z=x.h[row*C+c]+r.h[row*C+c]; ss+=double(z)*z;} for(int c=0;c<C;++c) o.ref[row*C+c]=float((x.h[row*C+c]+r.h[row*C+c])/std::sqrt(ss/C+eps)*w.h[c]);}
            });
            if(!f.run([&] { solve(x.data(), r.data(), w.data(), o.data(), N, C, eps); })) ++failed;
        }
        return failed ? 1 : 0;
    }, std::size(cases), list_solution_versions, select_solution_version);
}
