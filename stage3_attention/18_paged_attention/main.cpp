#include "solution.h"
#include "bench.hpp"
#include "reference.hpp"

using namespace bench;
namespace {
struct TestCase { const char* tier; int batch_size; int num_heads; int head_dim; int block_size; int max_blocks_per_seq; const char* pattern; };
const TestCase cases[] = {
    {"tiny", 1, 1, 4, 1, 1, "random"},
    {"small", 3, 2, 12, 4, 3, "random"},
    {"small", 2, 3, 32, 16, 5, "random"},
    {"medium", 4, 8, 64, 16, 33, "random"},
    {"large", 8, 16, 64, 16, 128, "random"}
};
}

int main(int argc, char** argv) {
    return entry(argc, argv, "p18_paged_attention", [&](const Options& options) {
        int failed=0, index=0;
        for(const auto& tc : cases) {
            int case_index=index++;
            if(!options.select(tc.tier,case_index)) continue;
            std::string label="batch_size=" + std::to_string(tc.batch_size) + ", " + "num_heads=" + std::to_string(tc.num_heads) + ", " + "head_dim=" + std::to_string(tc.head_dim) + ", " + "block_size=" + std::to_string(tc.block_size) + ", " + "max_blocks_per_seq=" + std::to_string(tc.max_blocks_per_seq) + ", pattern=" + tc.pattern;
            if(options.list) { std::cout << case_index << " [" << tc.tier << "] " << label << '\n'; continue; }
            const int batch_size=tc.batch_size; const int num_heads=tc.num_heads; const int head_dim=tc.head_dim; const int block_size=tc.block_size; const int max_blocks_per_seq=tc.max_blocks_per_seq;
            const std::string pattern=tc.pattern;
            std::mt19937 rng(options.seed + 18u*1009u + unsigned(case_index)*9176u);
            Fixture f(options,tc.tier,label,1e-4,1e-4);
            int B=batch_size,H=num_heads,D=head_dim,BS=block_size,NB=max_blocks_per_seq,P=B*NB+7;
            auto& q=f.input<float>("Q",size_t(B)*H*D); auto& k=f.input<float>("K_cache",size_t(P)*BS*H*D); auto& v=f.input<float>("V_cache",k.h.size()); auto& bt=f.input<int>("block_table",B*NB); auto& lens=f.input<int>("context_lens",B); auto& o=f.output<float>("output",q.h.size());
            fill(q.h,rng,"random"); fill(k.h,rng,"random"); fill(v.h,rng,"random");
            std::vector<int> pages(P); std::iota(pages.begin(),pages.end(),0); std::shuffle(pages.begin(),pages.end(),rng);
            for(int b=0;b<B;++b) {lens.h[b]=b==0 ? BS*NB : std::max(1,BS*NB-b*BS-1); for(int p=0;p<NB;++p) bt.h[b*NB+p]=pages[b*NB+p];}
            f.reference([&] {
                for(int b=0;b<B;++b) for(int h=0;h<H;++h) {auto idx=[&](int s,int j){return ((size_t(bt.h[b*NB+s/BS])*BS+s%BS)*H+h)*D+j;}; auto y=reference::attention_row(lens.h[b],D,[&](int j){return q.h[(b*H+h)*D+j];},[&](int s,int j){return k.h[idx(s,j)];},[&](int s,int j){return v.h[idx(s,j)];}); for(int j=0;j<D;++j) o.ref[(b*H+h)*D+j]=float(y[j]);}
            });
            if(!f.run([&] { solve(q.data(), k.data(), v.data(), bt.data(), lens.data(), o.data(), batch_size, num_heads, head_dim, block_size, max_blocks_per_seq); })) ++failed;
        }
        return failed ? 1 : 0;
    }, std::size(cases));
}
