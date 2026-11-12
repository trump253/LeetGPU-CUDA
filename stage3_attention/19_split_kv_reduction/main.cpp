#include "solution.h"
#include "bench.hpp"
#include "reference.hpp"

using namespace bench;
namespace {
struct TestCase { const char* tier; int num_splits; int num_heads; int head_dim; const char* pattern; };
const TestCase cases[] = {
    {"tiny", 1, 1, 1, "random"},
    {"small", 3, 2, 7, "equal"},
    {"small", 17, 3, 33, "extreme"},
    {"medium", 64, 32, 64, "random"},
    {"large", 128, 64, 128, "extreme"}
};
}

int main(int argc, char** argv) {
    return entry(argc, argv, "p19_split_kv_reduction", [&](const Options& options) {
        int failed=0, index=0;
        for(const auto& tc : cases) {
            int case_index=index++;
            if(!options.select(tc.tier,case_index)) continue;
            std::string label="num_splits=" + std::to_string(tc.num_splits) + ", " + "num_heads=" + std::to_string(tc.num_heads) + ", " + "head_dim=" + std::to_string(tc.head_dim) + ", pattern=" + tc.pattern;
            if(options.list) { std::cout << case_index << " [" << tc.tier << "] " << label << '\n'; continue; }
            const int num_splits=tc.num_splits; const int num_heads=tc.num_heads; const int head_dim=tc.head_dim;
            const std::string pattern=tc.pattern;
            std::mt19937 rng(options.seed + 19u*1009u + unsigned(case_index)*9176u);
            Fixture f(options,tc.tier,label,1e-4,1e-4);
            int S=num_splits,H=num_heads,D=head_dim;
            auto& p=f.input<float>("partial_out",size_t(S)*H*D); auto& l=f.input<float>("partial_lse",S*H); auto& o=f.output<float>("output",H*D); fill(p.h,rng,"random"); fill(l.h,rng,pattern=="equal" ? "zeros" : pattern);
            f.reference([&] {
                for(int h=0;h<H;++h) {std::vector<double> a(S); for(int s=0;s<S;++s) a[s]=l.h[s*H+h]; auto w=reference::softmax(a); for(int j=0;j<D;++j) {double sum=0; for(int s=0;s<S;++s) sum+=w[s]*p.h[(s*H+h)*D+j]; o.ref[h*D+j]=float(sum);}}
            });
            if(!f.run([&] { solve(p.data(), l.data(), o.data(), num_splits, num_heads, head_dim); })) ++failed;
        }
        return failed ? 1 : 0;
    }, std::size(cases), list_solution_versions, select_solution_version);
}
