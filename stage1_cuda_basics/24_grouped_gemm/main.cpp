#include "solution.h"
#include "bench.hpp"
#include "reference.hpp"

using namespace bench;
namespace {
struct TestCase { const char* tier; int G; int M_total; int K; int N; const char* pattern; };
const TestCase cases[] = {
    {"tiny", 1, 1, 1, 1, "random"},
    {"small", 4, 7, 17, 31, "empty"},
    {"small", 7, 33, 31, 17, "empty"},
    {"medium", 8, 256, 128, 257, "random"},
    {"large", 16, 1024, 256, 256, "empty"}
};
}

int main(int argc, char** argv) {
    return entry(argc, argv, "p24_grouped_gemm", [&](const Options& options) {
        int failed=0, index=0;
        for(const auto& tc : cases) {
            int case_index=index++;
            if(!options.select(tc.tier,case_index)) continue;
            std::string label="G=" + std::to_string(tc.G) + ", " + "M_total=" + std::to_string(tc.M_total) + ", " + "K=" + std::to_string(tc.K) + ", " + "N=" + std::to_string(tc.N) + ", pattern=" + tc.pattern;
            if(options.list) { std::cout << case_index << " [" << tc.tier << "] " << label << '\n'; continue; }
            const int G=tc.G; const int M_total=tc.M_total; const int K=tc.K; const int N=tc.N;
            const std::string pattern=tc.pattern;
            std::mt19937 rng(options.seed + 24u*1009u + unsigned(case_index)*9176u);
            Fixture f(options,tc.tier,label,3e-4,3e-4);
            auto& a=f.input<float>("A",size_t(M_total)*K); auto& b=f.input<float>("B",size_t(G)*K*N); auto& off=f.input<int>("group_offsets",G+1); auto& o=f.output<float>("C",size_t(M_total)*N); fill(a.h,rng,"random",0.25f); fill(b.h,rng,"random",0.25f);
            off.h[0]=0; off.h[G]=M_total; for(int g=1;g<G;++g) off.h[g]=int(rng()%(M_total+1)); std::sort(off.h.begin(),off.h.end()); if(pattern=="empty" && G>1) off.h[1]=0;
            f.reference([&] {
                for(int g=0;g<G;++g) for(int r=off.h[g];r<off.h[g+1];++r) for(int c=0;c<N;++c) {double s=0; for(int k=0;k<K;++k) s+=double(a.h[r*K+k])*b.h[(g*K+k)*N+c]; o.ref[r*N+c]=float(s);}
            });
            if(!f.run([&] { solve(a.data(), b.data(), off.data(), o.data(), G, M_total, K, N); })) ++failed;
        }
        return failed ? 1 : 0;
    }, std::size(cases), list_solution_versions, select_solution_version);
}
