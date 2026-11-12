#include "solution.h"
#include "bench.hpp"
#include "reference.hpp"

using namespace bench;
namespace {
struct TestCase { const char* tier; int T; int S; int H; int D; const char* pattern; };
const TestCase cases[] = {
    {"tiny", 1, 1, 1, 1, "random"},
    {"small", 2, 33, 3, 7, "random"},
    {"small", 1, 129, 2, 32, "large_scores"},
    {"medium", 2, 4097, 8, 64, "random"},
    {"large", 1, 32768, 16, 64, "random"}
};
}

int main(int argc, char** argv) {
    return entry(argc, argv, "p20_flash_decoding", [&](const Options& options) {
        int failed=0, index=0;
        for(const auto& tc : cases) {
            int case_index=index++;
            if(!options.select(tc.tier,case_index)) continue;
            std::string label="T=" + std::to_string(tc.T) + ", " + "S=" + std::to_string(tc.S) + ", " + "H=" + std::to_string(tc.H) + ", " + "D=" + std::to_string(tc.D) + ", pattern=" + tc.pattern;
            if(options.list) { std::cout << case_index << " [" << tc.tier << "] " << label << '\n'; continue; }
            const int T=tc.T; const int S=tc.S; const int H=tc.H; const int D=tc.D;
            const std::string pattern=tc.pattern;
            std::mt19937 rng(options.seed + 20u*1009u + unsigned(case_index)*9176u);
            Fixture f(options,tc.tier,label,1e-4,1e-4);
            auto& q=f.input<float>("Q",size_t(T)*H*D); auto& k=f.input<float>("K",size_t(S)*H*D); auto& v=f.input<float>("V",k.h.size()); auto& o=f.output<float>("output",q.h.size()); fill(q.h,rng,"random",pattern=="large_scores" ? 6.f : 1.f); fill(k.h,rng,"random",pattern=="large_scores" ? 6.f : 1.f); fill(v.h,rng,"random");
            f.reference([&] {
                for(int t=0;t<T;++t) for(int h=0;h<H;++h) {auto y=reference::attention_row(S,D,[&](int j){return q.h[(t*H+h)*D+j];},[&](int s,int j){return k.h[(s*H+h)*D+j];},[&](int s,int j){return v.h[(s*H+h)*D+j];}); for(int j=0;j<D;++j) o.ref[(t*H+h)*D+j]=float(y[j]);}
            });
            if(!f.run([&] { solve(q.data(), k.data(), v.data(), o.data(), T, S, H, D); })) ++failed;
        }
        return failed ? 1 : 0;
    }, std::size(cases));
}
