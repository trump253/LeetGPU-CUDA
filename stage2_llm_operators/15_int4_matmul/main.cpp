#include "solution.h"
#include "bench.hpp"
#include "reference.hpp"

using namespace bench;
namespace {
struct TestCase { const char* tier; int M; int N; int K; int group_size; const char* pattern; };
const TestCase cases[] = {
    {"tiny", 1, 3, 4, 2, "random"},
    {"small", 3, 17, 32, 8, "random"},
    {"small", 7, 33, 128, 128, "random"},
    {"medium", 32, 128, 256, 32, "random"},
    {"large", 128, 512, 512, 128, "random"}
};
}

int main(int argc, char** argv) {
    return entry(argc, argv, "p15_int4_matmul", [&](const Options& options) {
        int failed=0, index=0;
        for(const auto& tc : cases) {
            int case_index=index++;
            if(!options.select(tc.tier,case_index)) continue;
            std::string label="M=" + std::to_string(tc.M) + ", " + "N=" + std::to_string(tc.N) + ", " + "K=" + std::to_string(tc.K) + ", " + "group_size=" + std::to_string(tc.group_size) + ", pattern=" + tc.pattern;
            if(options.list) { std::cout << case_index << " [" << tc.tier << "] " << label << '\n'; continue; }
            const int M=tc.M; const int N=tc.N; const int K=tc.K; const int group_size=tc.group_size;
            const std::string pattern=tc.pattern;
            std::mt19937 rng(options.seed + 15u*1009u + unsigned(case_index)*9176u);
            Fixture f(options,tc.tier,label,2e-3,2e-3);
            auto& x=f.input<half>("x",size_t(M)*K); auto& w=f.input<uint8_t>("w_q",size_t(N)*K/2); auto& s=f.input<half>("scales",size_t(N)*(K/group_size)); auto& o=f.output<half>("y",size_t(M)*N);
            fill(x.h,rng,pattern,0.25f); for(auto& v:w.h) v=uint8_t(rng()%256); for(auto& v:s.h) v=__float2half(0.01f+float(rng()%100)/1000.f);
            f.reference([&] {
                for(int r=0;r<M;++r) for(int n=0;n<N;++n) {double sum=0; for(int k=0;k<K;++k) {int byte=w.h[n*(K/2)+k/2]; int q=((k%2==0 ? byte>>4 : byte)&15)-8; double weight=q*value(s.h[n*(K/group_size)+k/group_size]); sum+=value(x.h[r*K+k])*weight;} o.ref[r*N+n]=__float2half(float(sum));}
            });
            if(!f.run([&] { solve(x.data(), w.data(), s.data(), o.data(), M, N, K, group_size); })) ++failed;
        }
        return failed ? 1 : 0;
    }, std::size(cases), list_solution_versions, select_solution_version);
}
