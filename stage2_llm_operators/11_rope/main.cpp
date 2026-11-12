#include "solution.h"
#include "bench.hpp"
#include "reference.hpp"

using namespace bench;
namespace {
struct TestCase { const char* tier; int M; int D; const char* pattern; };
const TestCase cases[] = {
    {"tiny", 1, 2, "random"},
    {"small", 3, 6, "random"},
    {"small", 17, 130, "random"},
    {"medium", 1024, 128, "random"},
    {"large", 8192, 128, "random"}
};
}

int main(int argc, char** argv) {
    return entry(argc, argv, "p11_rope", [&](const Options& options) {
        int failed=0, index=0;
        for(const auto& tc : cases) {
            int case_index=index++;
            if(!options.select(tc.tier,case_index)) continue;
            std::string label="M=" + std::to_string(tc.M) + ", " + "D=" + std::to_string(tc.D) + ", pattern=" + tc.pattern;
            if(options.list) { std::cout << case_index << " [" << tc.tier << "] " << label << '\n'; continue; }
            const int M=tc.M; const int D=tc.D;
            const std::string pattern=tc.pattern;
            std::mt19937 rng(options.seed + 11u*1009u + unsigned(case_index)*9176u);
            Fixture f(options,tc.tier,label,1e-4,1e-4);
            auto& q=f.input<float>("Q",size_t(M)*D); auto& cs=f.input<float>("cos",q.h.size()); auto& sn=f.input<float>("sin",q.h.size()); auto& o=f.output<float>("output",q.h.size()); fill(q.h,rng,pattern);
            for(int r=0;r<M;++r) for(int j=0;j<D/2;++j) {double theta=r*std::pow(10000.,-2.*j/D); cs.h[r*D+j]=cs.h[r*D+j+D/2]=float(std::cos(theta)); sn.h[r*D+j]=sn.h[r*D+j+D/2]=float(std::sin(theta));}
            f.reference([&] {
                for(int r=0;r<M;++r) for(int j=0;j<D;++j) {int partner=(j+D/2)%D; float sign=j<D/2 ? -1.f : 1.f; o.ref[r*D+j]=q.h[r*D+j]*cs.h[r*D+j]+sign*q.h[r*D+partner]*sn.h[r*D+j];}
            });
            if(!f.run([&] { solve(q.data(), cs.data(), sn.data(), o.data(), M, D); })) ++failed;
        }
        return failed ? 1 : 0;
    }, std::size(cases));
}
