#include "solution.h"
#include "bench.hpp"
#include "reference.hpp"

using namespace bench;
namespace {
struct TestCase { const char* tier; int num_heads; int seq_len; int head_dim; const char* pattern; };
const TestCase cases[] = {
    {"tiny", 1, 1, 8, "random"},
    {"small", 3, 17, 24, "random"},
    {"small", 2, 129, 40, "random"},
    {"medium", 8, 1025, 64, "random"},
    {"large", 16, 8192, 64, "random"}
};
}

int main(int argc, char** argv) {
    return entry(argc, argv, "p23_int8_kv_attention", [&](const Options& options) {
        int failed=0, index=0;
        for(const auto& tc : cases) {
            int case_index=index++;
            if(!options.select(tc.tier,case_index)) continue;
            std::string label="num_heads=" + std::to_string(tc.num_heads) + ", " + "seq_len=" + std::to_string(tc.seq_len) + ", " + "head_dim=" + std::to_string(tc.head_dim) + ", pattern=" + tc.pattern;
            if(options.list) { std::cout << case_index << " [" << tc.tier << "] " << label << '\n'; continue; }
            const int num_heads=tc.num_heads; const int seq_len=tc.seq_len; const int head_dim=tc.head_dim;
            const std::string pattern=tc.pattern;
            std::mt19937 rng(options.seed + 23u*1009u + unsigned(case_index)*9176u);
            Fixture f(options,tc.tier,label,1e-4,1e-4);
            int H=num_heads,S=seq_len,D=head_dim;
            auto& q=f.input<float>("Q",H*D); auto& k=f.input<int8_t>("K_int8",size_t(H)*S*D); auto& v=f.input<int8_t>("V_int8",k.h.size()); auto& ks=f.input<float>("k_scale",H*S); auto& vs=f.input<float>("v_scale",H*S); auto& o=f.output<float>("output",q.h.size()); fill(q.h,rng,"random"); for(auto& a:k.h) a=int8_t(int(rng()%256)-128); for(auto& a:v.h) a=int8_t(int(rng()%256)-128); for(auto& a:ks.h) a=0.001f+float(rng()%50)*0.001f; for(auto& a:vs.h) a=0.001f+float(rng()%50)*0.001f;
            f.reference([&] {
                for(int h=0;h<H;++h) {auto y=reference::attention_row(S,D,[&](int j){return q.h[h*D+j];},[&](int s,int j){return float(k.h[(h*S+s)*D+j])*ks.h[h*S+s];},[&](int s,int j){return float(v.h[(h*S+s)*D+j])*vs.h[h*S+s];}); for(int j=0;j<D;++j) o.ref[h*D+j]=float(y[j]);}
            });
            if(!f.run([&] { solve(q.data(), k.data(), v.data(), ks.data(), vs.data(), o.data(), num_heads, seq_len, head_dim); })) ++failed;
        }
        return failed ? 1 : 0;
    }, std::size(cases), list_solution_versions, select_solution_version);
}
