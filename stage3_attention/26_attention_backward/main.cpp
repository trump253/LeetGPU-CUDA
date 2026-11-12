#include "solution.h"
#include "bench.hpp"
#include "reference.hpp"

using namespace bench;
namespace {
struct TestCase { const char* tier; int M; int N; int d; const char* pattern; };
const TestCase cases[] = {
    {"tiny", 1, 1, 1, "random"},
    {"small", 3, 5, 7, "random"},
    {"small", 17, 33, 31, "random"},
    {"medium", 64, 129, 32, "random"},
    {"large", 128, 512, 64, "random"}
};
}

int main(int argc, char** argv) {
    return entry(argc, argv, "p26_attention_backward", [&](const Options& options) {
        int failed=0, index=0;
        for(const auto& tc : cases) {
            int case_index=index++;
            if(!options.select(tc.tier,case_index)) continue;
            std::string label="M=" + std::to_string(tc.M) + ", " + "N=" + std::to_string(tc.N) + ", " + "d=" + std::to_string(tc.d) + ", pattern=" + tc.pattern;
            if(options.list) { std::cout << case_index << " [" << tc.tier << "] " << label << '\n'; continue; }
            const int M=tc.M; const int N=tc.N; const int d=tc.d;
            const std::string pattern=tc.pattern;
            std::mt19937 rng(options.seed + 26u*1009u + unsigned(case_index)*9176u);
            Fixture f(options,tc.tier,label,3e-4,3e-4);
            auto& q=f.input<float>("Q",size_t(M)*d); auto& k=f.input<float>("K",size_t(N)*d); auto& v=f.input<float>("V",k.h.size()); auto& dout=f.input<float>("dO",q.h.size()); auto& dq=f.output<float>("dQ",q.h.size()); auto& dk=f.output<float>("dK",k.h.size()); auto& dv=f.output<float>("dV",v.h.size()); fill(q.h,rng,"random"); fill(k.h,rng,"random"); fill(v.h,rng,"random"); fill(dout.h,rng,"random");
            f.reference([&] {
                reference::attention_backward(q.h,k.h,v.h,dout.h,M,N,d,dq.ref,dk.ref,dv.ref);
            });
            if(!f.run([&] { solve(q.data(), k.data(), v.data(), dout.data(), dq.data(), dk.data(), dv.data(), M, N, d); })) ++failed;
        }
        return failed ? 1 : 0;
    }, std::size(cases));
}
