#include "solution.h"
#include "bench.hpp"
#include "reference.hpp"

using namespace bench;
namespace {
struct TestCase { const char* tier; int M; int N; int K; const char* pattern; };
const TestCase cases[] = {
    {"tiny", 16, 16, 16, "random"},
    {"small", 17, 33, 19, "beta"},
    {"small", 32, 17, 65, "alpha0"},
    {"medium", 128, 192, 257, "beta"},
    {"large", 512, 512, 512, "beta"}
};
}

int main(int argc, char** argv) {
    return entry(argc, argv, "p07_gemm", [&](const Options& options) {
        int failed=0, index=0;
        for(const auto& tc : cases) {
            int case_index=index++;
            if(!options.select(tc.tier,case_index)) continue;
            std::string label="M=" + std::to_string(tc.M) + ", " + "N=" + std::to_string(tc.N) + ", " + "K=" + std::to_string(tc.K) + ", pattern=" + tc.pattern;
            if(options.list) { std::cout << case_index << " [" << tc.tier << "] " << label << '\n'; continue; }
            const int M=tc.M; const int N=tc.N; const int K=tc.K;
            const std::string pattern=tc.pattern;
            std::mt19937 rng(options.seed + 7u*1009u + unsigned(case_index)*9176u);
            Fixture f(options,tc.tier,label,2e-3,2e-3);
            auto& a=f.input<half>("A",size_t(M)*K); auto& b=f.input<half>("B",size_t(K)*N); auto& o=f.inout<half>("C",size_t(M)*N);
            fill(a.h,rng,"random",0.25f); fill(b.h,rng,"random",0.25f); fill(o.h,rng,"random");
            float alpha=pattern=="alpha0" ? 0.f : 1.25f, beta=pattern=="random" ? 0.f : -0.5f;
            f.reference([&] {
                for(int r=0;r<M;++r) for(int c=0;c<N;++c) {double sum=0; for(int k=0;k<K;++k) sum+=double(value(a.h[r*K+k]))*value(b.h[k*N+c]); o.ref[r*N+c]=__float2half(float(alpha*sum+beta*value(o.h[r*N+c])));}
            });
            if(!f.run([&] { solve(a.data(), b.data(), o.data(), M, N, K, alpha, beta); })) ++failed;
        }
        return failed ? 1 : 0;
    }, std::size(cases));
}
