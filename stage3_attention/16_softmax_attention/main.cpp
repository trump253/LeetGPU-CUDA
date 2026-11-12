#include "solution.h"
#include "bench.hpp"
#include "reference.hpp"

using namespace bench;
namespace {
struct TestCase { const char* tier; int M; int N; int d; const char* pattern; };
const TestCase cases[] = {
    {"tiny", 1, 1, 1, "random"},
    {"small", 3, 5, 7, "zeros"},
    {"small", 17, 33, 31, "random"},
    {"medium", 64, 257, 64, "random"},
    {"large", 256, 1024, 64, "random"}
};
}

int main(int argc, char** argv) {
    return entry(argc, argv, "p16_softmax_attention", [&](const Options& options) {
        int failed=0, index=0;
        for(const auto& tc : cases) {
            int case_index=index++;
            if(!options.select(tc.tier,case_index)) continue;
            std::string label="M=" + std::to_string(tc.M) + ", " + "N=" + std::to_string(tc.N) + ", " + "d=" + std::to_string(tc.d) + ", pattern=" + tc.pattern;
            if(options.list) { std::cout << case_index << " [" << tc.tier << "] " << label << '\n'; continue; }
            const int M=tc.M; const int N=tc.N; const int d=tc.d;
            const std::string pattern=tc.pattern;
            std::mt19937 rng(options.seed + 16u*1009u + unsigned(case_index)*9176u);
            Fixture f(options,tc.tier,label,1e-4,1e-4);
            auto& q=f.input<float>("Q",size_t(M)*d); auto& k=f.input<float>("K",size_t(N)*d); auto& v=f.input<float>("V",size_t(N)*d); auto& o=f.output<float>("output",q.h.size()); fill(q.h,rng,pattern); fill(k.h,rng,"random"); fill(v.h,rng,"random");
            f.reference([&] {
                for(int r=0;r<M;++r) {auto y=reference::attention_row(N,d,[&](int j){return q.h[r*d+j];},[&](int s,int j){return k.h[s*d+j];},[&](int s,int j){return v.h[s*d+j];}); for(int j=0;j<d;++j) o.ref[r*d+j]=float(y[j]);}
            });
            if(!f.run([&] { solve(q.data(), k.data(), v.data(), o.data(), M, N, d); })) ++failed;
        }
        return failed ? 1 : 0;
    }, std::size(cases));
}
