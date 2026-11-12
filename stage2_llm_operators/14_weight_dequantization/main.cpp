#include "solution.h"
#include "bench.hpp"
#include "reference.hpp"

using namespace bench;
namespace {
struct TestCase { const char* tier; int M; int N; int TILE_SIZE; const char* pattern; };
const TestCase cases[] = {
    {"tiny", 1, 1, 16, "random"},
    {"small", 17, 33, 16, "random"},
    {"small", 65, 31, 32, "random"},
    {"medium", 513, 1025, 64, "random"},
    {"large", 2048, 4097, 128, "random"}
};
}

int main(int argc, char** argv) {
    return entry(argc, argv, "p14_weight_dequantization", [&](const Options& options) {
        int failed=0, index=0;
        for(const auto& tc : cases) {
            int case_index=index++;
            if(!options.select(tc.tier,case_index)) continue;
            std::string label="M=" + std::to_string(tc.M) + ", " + "N=" + std::to_string(tc.N) + ", " + "TILE_SIZE=" + std::to_string(tc.TILE_SIZE) + ", pattern=" + tc.pattern;
            if(options.list) { std::cout << case_index << " [" << tc.tier << "] " << label << '\n'; continue; }
            const int M=tc.M; const int N=tc.N; const int TILE_SIZE=tc.TILE_SIZE;
            const std::string pattern=tc.pattern;
            std::mt19937 rng(options.seed + 14u*1009u + unsigned(case_index)*9176u);
            Fixture f(options,tc.tier,label,1e-4,1e-4);
            int sr=(M+TILE_SIZE-1)/TILE_SIZE,sc=(N+TILE_SIZE-1)/TILE_SIZE;
            auto& x=f.input<float>("X",size_t(M)*N); auto& s=f.input<float>("S",sr*sc); auto& o=f.output<float>("Y",x.h.size()); fill(x.h,rng,pattern,8); fill(s.h,rng,pattern);
            f.reference([&] {
                for(int r=0;r<M;++r) for(int c=0;c<N;++c) o.ref[r*N+c]=x.h[r*N+c]*s.h[(r/TILE_SIZE)*sc+c/TILE_SIZE];
            });
            if(!f.run([&] { solve(x.data(), s.data(), o.data(), M, N, TILE_SIZE); })) ++failed;
        }
        return failed ? 1 : 0;
    }, std::size(cases), list_solution_versions, select_solution_version);
}
