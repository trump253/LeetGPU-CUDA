#include "solution.h"
#include "bench.hpp"
#include "reference.hpp"

using namespace bench;
namespace {
struct TestCase { const char* tier; int N; int num_bins; const char* pattern; };
const TestCase cases[] = {
    {"tiny", 1, 1, "hot"},
    {"small", 257, 17, "random"},
    {"small", 1003, 256, "hot"},
    {"medium", 65539, 257, "random"},
    {"large", 4194307, 256, "random"}
};
}

int main(int argc, char** argv) {
    return entry(argc, argv, "p06_histogramming", [&](const Options& options) {
        int failed=0, index=0;
        for(const auto& tc : cases) {
            int case_index=index++;
            if(!options.select(tc.tier,case_index)) continue;
            std::string label="N=" + std::to_string(tc.N) + ", " + "num_bins=" + std::to_string(tc.num_bins) + ", pattern=" + tc.pattern;
            if(options.list) { std::cout << case_index << " [" << tc.tier << "] " << label << '\n'; continue; }
            const int N=tc.N; const int num_bins=tc.num_bins;
            const std::string pattern=tc.pattern;
            std::mt19937 rng(options.seed + 6u*1009u + unsigned(case_index)*9176u);
            Fixture f(options,tc.tier,label,0,0);
            auto& x=f.input<int>("input",N); auto& o=f.output<int>("histogram",num_bins);
            for(int j=0;j<N;++j) x.h[j]=pattern=="hot" ? num_bins/2 : int(rng()%num_bins);
            f.reference([&] {
                std::fill(o.ref.begin(),o.ref.end(),0); for(int v:x.h) if(v>=0 && v<num_bins) ++o.ref[v];
            });
            if(!f.run([&] { solve(x.data(), o.data(), N, num_bins); })) ++failed;
        }
        return failed ? 1 : 0;
    }, std::size(cases), list_solution_versions, select_solution_version);
}
