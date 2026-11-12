#include "solution.h"
#include "bench.hpp"
#include "reference.hpp"

using namespace bench;
namespace {
struct TestCase { const char* tier; int rows; int cols; const char* pattern; };
const TestCase cases[] = {
    {"tiny", 1, 1, "random"},
    {"small", 1, 37, "random"},
    {"small", 37, 1, "random"},
    {"small", 31, 33, "random"},
    {"medium", 257, 509, "random"},
    {"large", 2048, 4097, "random"}
};
}

int main(int argc, char** argv) {
    return entry(argc, argv, "p02_matrix_transpose", [&](const Options& options) {
        int failed=0, index=0;
        for(const auto& tc : cases) {
            int case_index=index++;
            if(!options.select(tc.tier,case_index)) continue;
            std::string label="rows=" + std::to_string(tc.rows) + ", " + "cols=" + std::to_string(tc.cols) + ", pattern=" + tc.pattern;
            if(options.list) { std::cout << case_index << " [" << tc.tier << "] " << label << '\n'; continue; }
            const int rows=tc.rows; const int cols=tc.cols;
            const std::string pattern=tc.pattern;
            std::mt19937 rng(options.seed + 2u*1009u + unsigned(case_index)*9176u);
            Fixture f(options,tc.tier,label,0,0);
            auto& x=f.input<float>("input",size_t(rows)*cols); auto& o=f.output<float>("output",x.h.size()); fill(x.h,rng,pattern);
            f.reference([&] {
                for(int r=0;r<rows;++r) for(int c=0;c<cols;++c) o.ref[c*rows+r]=x.h[r*cols+c];
            });
            if(!f.run([&] { solve(x.data(), o.data(), rows, cols); })) ++failed;
        }
        return failed ? 1 : 0;
    }, std::size(cases), list_solution_versions, select_solution_version);
}
