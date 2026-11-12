#include "solution.h"
#include "bench.hpp"
#include "reference.hpp"

using namespace bench;
namespace {
struct TestCase { const char* tier; int num_q_heads; int num_kv_heads; int seq_len; int head_dim; const char* pattern; };
const TestCase cases[] = {
    {"tiny", 1, 1, 1, 8, "random"},
    {"small", 4, 1, 17, 24, "random"},
    {"small", 6, 2, 33, 40, "random"},
    {"medium", 8, 2, 128, 64, "random"},
    {"large", 8, 2, 512, 64, "random"}
};
}

int main(int argc, char** argv) {
    return entry(argc, argv, "p17_grouped_query_attention", [&](const Options& options) {
        int failed=0, index=0;
        for(const auto& tc : cases) {
            int case_index=index++;
            if(!options.select(tc.tier,case_index)) continue;
            std::string label="num_q_heads=" + std::to_string(tc.num_q_heads) + ", " + "num_kv_heads=" + std::to_string(tc.num_kv_heads) + ", " + "seq_len=" + std::to_string(tc.seq_len) + ", " + "head_dim=" + std::to_string(tc.head_dim) + ", pattern=" + tc.pattern;
            if(options.list) { std::cout << case_index << " [" << tc.tier << "] " << label << '\n'; continue; }
            const int num_q_heads=tc.num_q_heads; const int num_kv_heads=tc.num_kv_heads; const int seq_len=tc.seq_len; const int head_dim=tc.head_dim;
            const std::string pattern=tc.pattern;
            std::mt19937 rng(options.seed + 17u*1009u + unsigned(case_index)*9176u);
            Fixture f(options,tc.tier,label,1e-4,1e-4);
            int Hq=num_q_heads,Hkv=num_kv_heads,S=seq_len,D=head_dim;
            auto& q=f.input<float>("Q",size_t(Hq)*S*D); auto& k=f.input<float>("K",size_t(Hkv)*S*D); auto& v=f.input<float>("V",k.h.size()); auto& o=f.output<float>("output",q.h.size()); fill(q.h,rng,pattern); fill(k.h,rng,pattern); fill(v.h,rng,pattern);
            f.reference([&] {
                for(int h=0;h<Hq;++h) for(int r=0;r<S;++r) {int kh=h/(Hq/Hkv); auto y=reference::attention_row(S,D,[&](int j){return q.h[(h*S+r)*D+j];},[&](int s,int j){return k.h[(kh*S+s)*D+j];},[&](int s,int j){return v.h[(kh*S+s)*D+j];}); for(int j=0;j<D;++j) o.ref[(h*S+r)*D+j]=float(y[j]);}
            });
            if(!f.run([&] { solve(q.data(), k.data(), v.data(), o.data(), num_q_heads, num_kv_heads, seq_len, head_dim); })) ++failed;
        }
        return failed ? 1 : 0;
    }, std::size(cases));
}
