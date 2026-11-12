#include "solution.h"
#include "bench.hpp"
#include "reference.hpp"

using namespace bench;
namespace {
struct TestCase { const char* tier; int B; int d_model; int H_q; int H_kv; int D; int S_max; const char* pattern; };
const TestCase cases[] = {
    {"tiny", 1, 2, 1, 1, 2, 2, "random"},
    {"small", 3, 17, 4, 2, 6, 7, "random"},
    {"medium", 8, 128, 8, 2, 32, 64, "random"},
    {"large", 16, 512, 8, 2, 64, 128, "random"}
};
}

int main(int argc, char** argv) {
    return entry(argc, argv, "p22_fused_qkv_rope_cache", [&](const Options& options) {
        int failed=0, index=0;
        for(const auto& tc : cases) {
            int case_index=index++;
            if(!options.select(tc.tier,case_index)) continue;
            std::string label="B=" + std::to_string(tc.B) + ", " + "d_model=" + std::to_string(tc.d_model) + ", " + "H_q=" + std::to_string(tc.H_q) + ", " + "H_kv=" + std::to_string(tc.H_kv) + ", " + "D=" + std::to_string(tc.D) + ", " + "S_max=" + std::to_string(tc.S_max) + ", pattern=" + tc.pattern;
            if(options.list) { std::cout << case_index << " [" << tc.tier << "] " << label << '\n'; continue; }
            const int B=tc.B; const int d_model=tc.d_model; const int H_q=tc.H_q; const int H_kv=tc.H_kv; const int D=tc.D; const int S_max=tc.S_max;
            const std::string pattern=tc.pattern;
            std::mt19937 rng(options.seed + 22u*1009u + unsigned(case_index)*9176u);
            Fixture f(options,tc.tier,label,3e-4,3e-4);
            int width=(H_q+2*H_kv)*D;
            auto& x=f.input<float>("x",size_t(B)*d_model); auto& w=f.input<float>("W_qkv",size_t(d_model)*width); auto& cs=f.input<float>("cos_sin_cache",S_max*D); auto& pos=f.input<int>("positions",B); auto& kc=f.inout<float>("K_cache",size_t(B)*H_kv*S_max*D); auto& vc=f.inout<float>("V_cache",kc.h.size()); auto& o=f.output<float>("Q_out",size_t(B)*H_q*D);
            fill(x.h,rng,"random"); fill(w.h,rng,"random",0.1f); fill(kc.h,rng,"random"); fill(vc.h,rng,"random");
            for(int b=0;b<B;++b) pos.h[b]=b==0 ? S_max-1 : (b==1 ? 0 : int(rng()%S_max));
            for(int p=0;p<S_max;++p) for(int j=0;j<D/2;++j) {double a=p*std::pow(10000.,-2.*j/D); cs.h[p*D+j]=float(std::cos(a)); cs.h[p*D+D/2+j]=float(std::sin(a));}
            f.reference([&] {
                kc.ref=kc.h; vc.ref=vc.h; std::vector<double> proj(size_t(B)*width);
                for(int b=0;b<B;++b) for(int j=0;j<width;++j) {double s=0; for(int a=0;a<d_model;++a) s+=double(x.h[b*d_model+a])*w.h[a*width+j]; proj[b*width+j]=s;}
                for(int b=0;b<B;++b) {int p=pos.h[b]; for(int h=0;h<H_q+H_kv;++h) for(int j=0;j<D/2;++j) {double a=proj[b*width+h*D+j],z=proj[b*width+h*D+j+D/2],c=cs.h[p*D+j],s=cs.h[p*D+j+D/2]; auto& dst=h<H_q ? o.ref : kc.ref; size_t base=h<H_q ? size_t(b*H_q+h)*D : ((size_t(b)*H_kv+h-H_q)*S_max+p)*D; dst[base+j]=float(a*c-z*s); dst[base+j+D/2]=float(z*c+a*s);} for(int h=0;h<H_kv;++h) for(int j=0;j<D;++j) vc.ref[((size_t(b)*H_kv+h)*S_max+p)*D+j]=float(proj[b*width+(H_q+H_kv+h)*D+j]);}
            });
            if(!f.run([&] { solve(x.data(), w.data(), cs.data(), pos.data(), kc.data(), vc.data(), o.data(), B, d_model, H_q, H_kv, D, S_max); })) ++failed;
        }
        return failed ? 1 : 0;
    }, std::size(cases));
}
