#include "solution.h"
#include "bench.hpp"
#include "reference.hpp"

using namespace bench;
namespace {
struct TestCase { const char* tier; int M; int E; int k; const char* pattern; };
const TestCase cases[] = {
    {"tiny", 1, 1, 1, "random"},
    {"small", 3, 17, 1, "random"},
    {"small", 7, 31, 31, "random"},
    {"medium", 128, 64, 2, "random"},
    {"large", 4096, 256, 8, "random"}
};
}

int main(int argc, char** argv) {
    return entry(argc, argv, "p21_moe_topk_gating", [&](const Options& options) {
        int failed=0, index=0;
        for(const auto& tc : cases) {
            int case_index=index++;
            if(!options.select(tc.tier,case_index)) continue;
            std::string label="M=" + std::to_string(tc.M) + ", " + "E=" + std::to_string(tc.E) + ", " + "k=" + std::to_string(tc.k) + ", pattern=" + tc.pattern;
            if(options.list) { std::cout << case_index << " [" << tc.tier << "] " << label << '\n'; continue; }
            const int M=tc.M; const int E=tc.E; const int k=tc.k;
            const std::string pattern=tc.pattern;
            std::mt19937 rng(options.seed + 21u*1009u + unsigned(case_index)*9176u);
            Fixture f(options,tc.tier,label,1e-4,1e-4);
            auto& x=f.input<float>("logits",size_t(M)*E); auto& w=f.output<float>("topk_weights",size_t(M)*k); auto& ids=f.output<int>("topk_indices",size_t(M)*k);
            std::vector<int> order(E); std::iota(order.begin(),order.end(),0); for(int r=0;r<M;++r) {std::shuffle(order.begin(),order.end(),rng); for(int j=0;j<E;++j) x.h[r*E+j]=(order[j]-E/2)*0.125f;}
            f.reference([&] {
                for(int r=0;r<M;++r) {std::vector<int> ix(E); std::iota(ix.begin(),ix.end(),0); std::partial_sort(ix.begin(),ix.begin()+k,ix.end(),[&](int a,int b){return x.h[r*E+a]>x.h[r*E+b];}); std::vector<double> vals(k); for(int j=0;j<k;++j) {ids.ref[r*k+j]=ix[j]; vals[j]=x.h[r*E+ix[j]];} auto p=reference::softmax(vals); for(int j=0;j<k;++j) w.ref[r*k+j]=float(p[j]);}
            });
            if(!f.run([&] { solve(x.data(), w.data(), ids.data(), M, E, k); })) ++failed;
        }
        return failed ? 1 : 0;
    }, std::size(cases));
}
