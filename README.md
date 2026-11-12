# LeetGPU 本地 CUDA 刷题环境

按 [LeetGPU.md](LeetGPU.md) 的三个阶段组织，包含全部 **21 道核心题 + 5 道选做题**。每题有独立的 `solution.cu`、接口头文件、测试主函数、中文 README 和 CMake target。

每题支持多个实现版本，`solve()` 对外接口保持不变。第 01/02 题已保留你完成的实现并注册为独立版本；其余题目提供待填写的 `v0: starter`。测试数据、CPU 参考结果、GPU 内存管理、正确性检查和计时已经准备好；空 kernel 仍会显示 `FAIL`。`--self-test` 仅检查测试链路，不代表题目已做完。

## 立即开始

在此仓库根目录执行：

```bash
# 使用本机 CMake 和 /usr/local/cuda/bin/nvcc
cmake --preset release
# 可只编译一道题
cmake --build --preset release --target p02_matrix_transpose -j 4

# 修改 stage1_cuda_basics/02_matrix_transpose/solution.cu 后运行
./build/release/bin/p02_matrix_transpose

# 也可以按题号自动配置、编译和运行
python3 scripts/run.py 02
python3 scripts/run.py 02 -- --size small --iterations 20
python3 scripts/run.py 02 -- --list-versions
python3 scripts/run.py 02 -- --version all --size small --iterations 20
```

查看并验证整个脚手架：

```bash
python3 scripts/run.py --list
cmake --build --preset release -j 4
ctest --preset scaffold
```

验证自己写的答案：

```bash
./build/release/bin/p02_matrix_transpose --check-only
# 只测试某一道解答；未实现时应失败
ctest --test-dir build/release -R '^solution_p02_' --output-on-failure
# 全部解答；尚未完成的题会失败
ctest --preset solutions
```

不要把 `ctest --preset scaffold` 的通过数当作已解题数。不加过滤直接运行 `ctest --test-dir build/release` 会同时运行 scaffold 和 solution 两类测试。

## 目录与修改位置

```text
stage1_cuda_basics/             # 01–07，选做 24/25
stage2_llm_operators/           # 08–15
stage3_attention/              # 16–21，选做 22/23/26
  16_softmax_attention/
    solution.cu               # kernel、各版 solve_vN、版本表和 extern "C" solve
    solution.h                # 官方 solve 接口声明及公共版本 API
    main.cpp                  # 用例、随机数据、CPU 参考、main
    README.md                 # 题意、布局、规模、编译运行命令
    CMakeLists.txt
common/bench.hpp               # 公共显存、校验、CLI、计时
common/solution_versions.hpp    # 版本表校验、类型安全的 launcher 查找和选择 API
common/reference.hpp           # CPU softmax / attention 数学参考
scripts/run.py                 # 按题号/路径运行，无第三方 Python 依赖
scripts/problems.json          # 题目索引
.vscode/                       # IntelliSense / 构建任务 / cuda-gdb
build/release/bin/             # 编译产物
```

为保留三个阶段目录，选做的 Grouped GEMM、Top K Selection 放在第一阶段；融合 QKV、INT8 KV Attention、Attention Backward 放在第三阶段。编号与原题单一致。

## 多版本实现

每个版本由一个普通 CPU 函数 `solve_vN()` 管理自己的 grid、block、shared memory 和一个或多个 kernel。`main.cpp` 在预热和计时之前选择版本，计时内的 `solve()` 只通过函数指针调用对应 launcher。

```bash
# 查看版本 ID、名称以及显式指定的默认版本，无需 GPU
python3 scripts/run.py 02 -- --list-versions
# ID 和名称都可选择
python3 scripts/run.py 02 -- --version 0 --size small
python3 scripts/run.py 02 -- --version tiled16_padding --size small
# 同一进程内，按注册顺序分别校验和计时所有版本
python3 scripts/run.py 02 -- --version all --size small --warmup 5 --iterations 30
# 不指定 --version，或指定 default，使用 kDefaultVersion
python3 scripts/run.py 02 -- --version default --check-only
```

当前已有版本：

| 题目 | v0 | v1 | 默认 |
| --- | --- | --- | --- |
| 01 Vector Addition | `scalar`，复用你现有的标量 kernel | `float4_tail`，保留 float4 加尾部处理 | 1 |
| 02 Matrix Transpose | `naive`，现有朴素转置 | `tiled16_padding`，现有 16×17 shared tile | 1 |
| 其他题目 | `starter`，待完成的原始模板 | 尚未注册 | 0 |

新增版本只需修改对应题目的 `solution.cu`：

1. 在匿名 namespace 中新增与 `solve()` 参数一致的 `solve_v2()`，编写其 host 启动逻辑。GPU kernel 仍定义在文件前部；不同版本可以复用已有 kernel。
2. 在 `kVersions` 中增加 `{2, "your_version_name", solve_v2}`。名称不得重复、不得以数字开头，也不能使用保留名称 `all`、`default`；ID 是唯一的非负整数，允许不连续。
3. 如需设为默认，将 `kDefaultVersion` 改成 `2`。新增版本不会自动改变默认选择。

例如，转置题的版本表可扩展为以下结构（先自行完成 `solve_v2`）：

```cpp
using SolveFn = decltype(&solve);
constexpr solution::Version<SolveFn> kVersions[] = {
    {0, "naive", solve_v0},
    {1, "tiled16_padding", solve_v1},
    {2, "your_version_name", solve_v2},
};
constexpr int kDefaultVersion = 1;
```

其余注册、选择和 `solve()` 转发代码可直接沿用；`main.cpp`、CMake 和 VS Code 配置不需要随版本增加而修改。版本 ID/名称重复、默认 ID 不存在、launcher 为空会在编译时拒绝；命令行指定不存在的版本会明确报错并以 2 退出，不能静默回退。`--version` 只在一次启动时选择，可以直接与 `--case`、`--size`、`--seed`、`--check-only`、`--self-test` 组合。

`--version all` 对每一版重新生成同样种子、同样尺寸的数据，并建立独立的 fixture；所有输入、GEMM 的 C_initial、可写 KV Cache 初始状态相同。每次 solve 调用前照常复位输出及可写输入。各版分别完成正确性检查、预热和计时，控制台标记 `[version=ID:NAME]`。错误结果返回 1，并继续检查其余版本；CUDA/运行时异常返回 2 并停止本次运行。

所有版本按注册顺序运行，不交替执行 A/B，也不主动清空缓存。性能仍可能受 GPU 频率和执行顺序影响；`--version all` 提供一致的输入与测量流程，不代表严格消除了所有性能波动。版本选择器使用进程内状态，当前脚手架按单线程、顺序运行设计。

## 测试数据与校验

共有 **134 组**自定义测试，覆盖 tiny/small/medium/large，具体尺寸见各题 README 或 `--list-cases`。数据在内存中按固定种子生成，无需 PyTorch、网络或数据文件。每题的 `main.cpp` 可直接增添用例。CPU 参考以清晰的标量公式为主，归约和 attention 使用 double 中间值；不会替你提供 GPU 算法。

测试包含非整齐尾部、矩形矩阵、全零/常量/抵消输入、较大 softmax score、FP16 GEMM 的非零 beta、INT4 nibble 顺序与分组、不同长度且物理页打乱的 KV Cache、可写 cache 其他槽的保留、空 GEMM 分组及 Top-K 边界。Moe Top-K 使用无重复 logit，避免未定义的并列索引顺序。

- 浮点逐元素验证 `abs(actual-reference) <= atol + rtol*abs(reference)`，各题容差写在 README 和 main.cpp 中；NaN/Inf 拒绝。
- 整数输出精确比较；只读输入按字节检查，防止误写。
- 纯输出先填 NaN/整数哨兵，漏写不会误判通过；每个设备分配前后有 256 字节保护区，用于检测附近的越界写。
- GEMM 的输入输出 C、QKV 的 K/V Cache 每次调用都恢复初始值；其他输出也会复位，不能依赖上次调用结果。
- 每个用例先独立校验两次，再预热、计时，计时结束再校验一次。错误用例跳过计时，退出码为 1；CUDA/参数错误退出码为 2。

`--self-test` 会把 CPU 参考结果上传到 GPU 再读回验证，确认数据、显存和比较链路可用；它不调用 solve，也不报告 kernel 性能。另有独立测试验证比较器能拒绝漏写、非有限数、越界写、输入误写，并以有限差分验证 attention 的 dQ/dK/dV 公式。

这里的规模是适合本地迭代的测试集，未复制官方的全部大型性能测试或隐藏测试。长上下文、大 GEMM 可在正确后逐步增大；完整 CPU 参考的成本也会随规模增加。

## 参数与计时口径

```bash
./build/release/bin/p02_matrix_transpose --help
./build/release/bin/p02_matrix_transpose --list-cases
./build/release/bin/p02_matrix_transpose --case 3 --check-only
./build/release/bin/p02_matrix_transpose --size large --warmup 5 --iterations 30
./build/release/bin/p02_matrix_transpose --seed 123 --device 1
```

`--size` 支持 `all`（默认）、`tiny`、`small`、`medium`、`large`；`--case` 从 0 编号，不与 `--size` 同用。`--version` 支持 ID、名称、`all` 和 `default`；默认使用各题显式指定的版本。`--list-cases` 只列一次用例，`--list-versions` 列出实现版本，这两种操作都不需要 GPU。默认 GPU 0、种子 42、预热 3 次、计时 10 次。`CUDA_VISIBLE_DEVICES` 也可用于选择 GPU。

正确后，每个规模输出以下字段：

```text
[version=1:tiled16_padding] [small] rows=31, cols=33, pattern=random | device_MiB=... | cpu_ref_ms=...
  PASS | gpu_us mean=... min=... median=... p95=... | solve_wall_us mean=...
```

`gpu_us` 由默认 stream 上的一对 CUDA event 记录，覆盖 solve 提交的全部 GPU 工作；`solve_wall_us` 是主机端调用 solve 并等待 event 完成的耗时。分配、输入拷贝、输出复位、CPU 参考、正确性验证均在计时区间之外。solve 内自己分配、清零、拷贝或同步的成本仍可能进入测量。微小 kernel 的 event 时间可能包含主机提交间隙，应结合最小值与 wall 时间判断。

请使用默认 stream；如果自己创建 stream，返回前须将其工作正确关联到默认 stream 或完成同步，否则默认 stream 的 event 无法代表全部工作。脚手架负责错误检查和同步，普通实现不必在 solve 中重复 `cudaDeviceSynchronize()`。

本地计时重复使用输入，不主动清空 L2，报告的是本地热缓存条件下的运行结果；不与在线排行榜数字直接等同。测性能使用 Release，Debug 的 `-G` 会改变 kernel 性能。

## VS Code

用 **Open Folder** 打开仓库根目录 `/root/leetgpu`，在远程侧使用 `ms-vscode.cpptools` 和 `ms-vscode.cmake-tools`（本机已安装）。

- `.cu/.cuh` 关联为 `cuda-cpp`，保留 CUDA 语法解析；没有将 `__global__` 等关键字伪装为空宏，也没有关闭错误提示。
- `c_cpp_properties.json` 使用 `/usr/local/cuda/bin/nvcc`、CUDA include 路径、C++17，以及实际生成的 `build/release/compile_commands.json`。主函数是普通 C++，其编译命令由数据库提供。
- CMake Tools 使用 presets 和本机 `/usr/local/bin/cmake`，首次打开可自动配置；手动选择 `release` configure preset 即可。
- 打开任意题目目录内的文件，`Ctrl+Shift+B` 构建当前题目；在 **Terminal → Run Task** 运行 `CUDA: run current exercise` 或 `CUDA: check current exercise (small)`。
- 使用 `CUDA: run current exercise (choose version)` 输入版本 ID、名称或 `all`；`CUDA: compare current exercise versions (small)` 对当前题目的所有注册版本进行 small 规模校验和计时。
- 保持当前编辑器文件位于题目目录，按 **F5** 使用 `cuda-gdb` 调试当前题目。启动前自动编译对应 Debug target，默认只跑 small 正确性测试。根目录文件不对应题目，不能用“当前题目”任务。
- 如需调试特定版本，在调试配置中选择 `CUDA: debug current exercise (choose version)` 并输入版本选择值；原有默认调试入口继续使用 `kDefaultVersion`。

如果之前打开过未配置状态，先执行 `cmake --preset release`，再执行 **C/C++: Reset IntelliSense Database** 或 **Developer: Reload Window** 刷新缓存。编译数据库重新生成即可跟随实际编译选项，无需手工填写 GCC 内部 include 目录。

## Debug 与内存检查

```bash
cmake --preset debug
cmake --build --preset debug --target p02_matrix_transpose -j 4
/usr/local/cuda/bin/cuda-gdb --args ./build/debug/bin/p02_matrix_transpose --case 3 --check-only

# 写好实现后检查越界访问、非法访问
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 2 \
  ./build/release/bin/p02_matrix_transpose --size small --check-only
# 检查 shared-memory 数据竞争
/usr/local/cuda/bin/compute-sanitizer --tool racecheck --error-exitcode 2 \
  ./build/release/bin/p02_matrix_transpose --size small --check-only
```

保护区只能捕获附近的越界写，不能替代 compute-sanitizer 的越界读/写和同步检查。

## 本机工具链与接口来源

搭建环境：CMake 4.2.3、CUDA 12.8.93、GCC 13.3.0、两张 RTX 3090（SM 8.6）。默认 `CMAKE_CUDA_ARCHITECTURES=native`；复制到其他机器或无 GPU 的编译机时，可显式指定：

```bash
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=86
```

最低 CMake 3.24，运行脚本只需要 Python 3 标准库。所有题目保持核对时的 [官方 CUDA starter 接口](https://github.com/AlphaGPU/leetgpu-challenges)，核对的 commit 记录在 [cmake/upstream.txt](cmake/upstream.txt)，每题 README 链接到该 commit 的题面。只保留接口约定；本地测试生成器、主函数和 CPU 参考独立编写。离线构建不依赖官方仓库。

特别注意：第 07 题为 FP16 alpha/beta GEMM；第 09 题为一维、标量 gamma/beta RMSNorm；第 14 题为 FP32 tile scale；第 15 题 INT4 高 nibble 存偶数位；第 17 题是全序列 GQA；第 22 题是连续布局的 KV Cache。各题 README 给出了准确布局。

## 题目索引

| 编号 | 阶段 | 题目 | CMake target |
| --- | --- | --- | --- |
| 01 | 1 | [Vector Addition](stage1_cuda_basics/01_vector_addition/README.md) | `p01_vector_addition` |
| 02 | 1 | [Matrix Transpose](stage1_cuda_basics/02_matrix_transpose/README.md) | `p02_matrix_transpose` |
| 03 | 1 | [Reduction](stage1_cuda_basics/03_reduction/README.md) | `p03_reduction` |
| 04 | 1 | [Dot Product](stage1_cuda_basics/04_dot_product/README.md) | `p04_dot_product` |
| 05 | 1 | [Prefix Sum](stage1_cuda_basics/05_prefix_sum/README.md) | `p05_prefix_sum` |
| 06 | 1 | [Histogramming](stage1_cuda_basics/06_histogramming/README.md) | `p06_histogramming` |
| 07 | 1 | [General Matrix Multiplication (GEMM)](stage1_cuda_basics/07_gemm/README.md) | `p07_gemm` |
| 08 | 2 | [Softmax](stage2_llm_operators/08_softmax/README.md) | `p08_softmax` |
| 09 | 2 | [RMS Normalization](stage2_llm_operators/09_rms_normalization/README.md) | `p09_rms_normalization` |
| 10 | 2 | [Layer Normalization](stage2_llm_operators/10_layer_normalization/README.md) | `p10_layer_normalization` |
| 11 | 2 | [Rotary Positional Embedding (RoPE)](stage2_llm_operators/11_rope/README.md) | `p11_rope` |
| 12 | 2 | [SwiGLU MLP Block](stage2_llm_operators/12_swiglu_mlp/README.md) | `p12_swiglu_mlp` |
| 13 | 2 | [Fused Residual Add and RMS Norm](stage2_llm_operators/13_fused_residual_rms_norm/README.md) | `p13_fused_residual_rms_norm` |
| 14 | 2 | [Weight Dequantization](stage2_llm_operators/14_weight_dequantization/README.md) | `p14_weight_dequantization` |
| 15 | 2 | [INT4 Weight-Only Quantized MatMul](stage2_llm_operators/15_int4_matmul/README.md) | `p15_int4_matmul` |
| 16 | 3 | [Softmax Attention](stage3_attention/16_softmax_attention/README.md) | `p16_softmax_attention` |
| 17 | 3 | [Grouped Query Attention (GQA)](stage3_attention/17_grouped_query_attention/README.md) | `p17_grouped_query_attention` |
| 18 | 3 | [Paged KV-Cache Attention](stage3_attention/18_paged_attention/README.md) | `p18_paged_attention` |
| 19 | 3 | [Split-KV Attention Reduction](stage3_attention/19_split_kv_reduction/README.md) | `p19_split_kv_reduction` |
| 20 | 3 | [Flash Decoding Attention](stage3_attention/20_flash_decoding/README.md) | `p20_flash_decoding` |
| 21 | 3 | [MoE Top-K Gating](stage3_attention/21_moe_topk_gating/README.md) | `p21_moe_topk_gating` |
| 22 | 3 | [Fused QKV Projection with RoPE and KV Cache Update（选做）](stage3_attention/22_fused_qkv_rope_cache/README.md) | `p22_fused_qkv_rope_cache` |
| 23 | 3 | [INT8 KV-Cache Attention（选做）](stage3_attention/23_int8_kv_attention/README.md) | `p23_int8_kv_attention` |
| 24 | 1 | [Grouped GEMM（选做）](stage1_cuda_basics/24_grouped_gemm/README.md) | `p24_grouped_gemm` |
| 25 | 1 | [Top K Selection（选做）](stage1_cuda_basics/25_top_k_selection/README.md) | `p25_top_k_selection` |
| 26 | 3 | [Softmax Attention Backward（选做）](stage3_attention/26_attention_backward/README.md) | `p26_attention_backward` |

## 初始搭建验收（2026-11-12）

- 26 个练习 target 全部通过 Release 编译；转置题通过 Debug 编译。
- `ctest --preset scaffold`：28/28 通过，包含全部 134 组数据的传输/校验自检，以及 CPU 数学参考和测试框架检查。
- 26 个空模板的最小用例均被正确判为 FAIL，不会被当作已实现答案。
- 公共框架通过 compute-sanitizer memcheck，0 errors；cuda-gdb 实际命中转置题的 GPU kernel 断点。
- VS Code JSON、C/C++ properties 和 CMake presets 通过本机已安装扩展的 schema 校验，全部题目源文件均有编译数据库条目。未通过 GUI 操作确认编辑器诊断状态。

初始自检输出在本地 `build/scaffold-validation.log`；构建目录不纳入版本控制。这些结果验证的是脚手架，题目 GPU 实现仍由你完成。

## 多版本改造验收（2026-11-12）

- 26 道题全部通过 Release 编译，共注册 28 个实现版本；第 01/02 题默认保留当前优化版，原有 kernel 源码逐字保留。
- `ctest --preset scaffold`：31/31 通过，题目自检覆盖 134 个用例、146 个用例/版本组合；新增测试覆盖版本表、默认选择、多 kernel launcher、可写状态复位，以及无 GPU 时的版本列表/参数错误处理。
- 第 01/02 题的全部已注册版本通过所有规模的正确性检查；small 规模的多版本计时输出正常。
- 第 01/02 题全部版本的 small 用例通过 compute-sanitizer memcheck，0 errors；转置全部版本的 small 用例通过 racecheck，0 hazards。
- 第 01/02 题 Debug 构建通过；VS Code 增加按版本运行、比较和调试入口。

本轮自检记录在 `build/version-validation.log`，内存/竞争检查记录在 `build/version-*-memcheck.log`、`build/version-transpose-racecheck.log`。其余 24 道题仍保留待填写的 `starter`，自检通过不代表这些题目已实现。
