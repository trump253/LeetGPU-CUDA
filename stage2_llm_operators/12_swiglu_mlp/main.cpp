#include "solution.h"
#include "bench.hpp"
#include "reference.hpp"

using namespace bench;
namespace {
struct TestCase { const char* tier; int M; int d_model; int d_ffn; const char* pattern; };
const TestCase cases[] = {
    {"tiny", 1, 2, 3, "random"},
    {"small", 3, 17, 31, "random"},
    {"medium", 32, 128, 257, "random"},
    {"large", 128, 256, 512, "random"}
};
}

int main(int argc, char** argv) {
    return entry(argc, argv, "p12_swiglu_mlp", [&](const Options& options) {
        int failed=0, index=0;
        for(const auto& tc : cases) {
            int case_index=index++;
            if(!options.select(tc.tier,case_index)) continue;
            std::string label="M=" + std::to_string(tc.M) + ", " + "d_model=" + std::to_string(tc.d_model) + ", " + "d_ffn=" + std::to_string(tc.d_ffn) + ", pattern=" + tc.pattern;
            if(options.list) { std::cout << case_index << " [" << tc.tier << "] " << label << '\n'; continue; }
            const int M=tc.M; const int d_model=tc.d_model; const int d_ffn=tc.d_ffn;
            const std::string pattern=tc.pattern;
            std::mt19937 rng(options.seed + 12u*1009u + unsigned(case_index)*9176u);
            Fixture f(options,tc.tier,label,3e-4,3e-4);
            auto& x=f.input<float>("x",size_t(M)*d_model); auto& g=f.input<float>("W_gate",size_t(d_model)*d_ffn); auto& u=f.input<float>("W_up",g.h.size()); auto& w=f.input<float>("W_down",size_t(d_ffn)*d_model); auto& o=f.output<float>("output",x.h.size());
            fill(x.h,rng,pattern); fill(g.h,rng,pattern,0.1f); fill(u.h,rng,pattern,0.1f); fill(w.h,rng,pattern,0.1f);
            f.reference([&] {
                std::vector<double> hidden(size_t(M)*d_ffn);
                for(int r=0;r<M;++r) for(int j=0;j<d_ffn;++j) {double a=0,b=0; for(int k=0;k<d_model;++k) {a+=double(x.h[r*d_model+k])*g.h[k*d_ffn+j]; b+=double(x.h[r*d_model+k])*u.h[k*d_ffn+j];} hidden[r*d_ffn+j]=a/(1+std::exp(-a))*b;}
                for(int r=0;r<M;++r) for(int j=0;j<d_model;++j) {double v=0; for(int k=0;k<d_ffn;++k) v+=hidden[r*d_ffn+k]*w.h[k*d_model+j]; o.ref[r*d_model+j]=float(v);}
            });
            if(!f.run([&] { solve(x.data(), g.data(), u.data(), w.data(), o.data(), M, d_model, d_ffn); })) ++failed;
        }
        return failed ? 1 : 0;
    }, std::size(cases));
}
