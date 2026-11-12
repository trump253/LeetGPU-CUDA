# 12. SwiGLU MLP Block

核心题。 [LeetGPU](https://leetgpu.com/challenges/swiglu-mlp-block) · [核对的官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/84_swiglu_mlp_block/challenge.html) · [总使用说明](../../README.md)

## 任务与接口

FP32 `x[M,d_model]`，`W_gate/W_up[d_model,d_ffn]`，`W_down[d_ffn,d_model]`。计算完整 MLP：`(silu(x@W_gate)*(x@W_up))@W_down`，其中 `silu(z)=z/(1+exp(-z))`。

所有指针都是 GPU device pointer，所有张量连续、行主序。只需实现 `solution.cu`；`solution.h` 固定接口，`main.cpp` 提供测试数据、CPU 参考结果与主函数。

```cpp
extern "C" void solve(const float* x, const float* W_gate, const float* W_up, const float* W_down,
                      float* output, int M, int d_model, int d_ffn);
```

## 本地测试数据

| 编号 | 规模 | 参数 | 数据 |
| --- | --- | --- | --- |
| 0 | tiny | M=1, d_model=2, d_ffn=3 | random |
| 1 | small | M=3, d_model=17, d_ffn=31 | random |
| 2 | medium | M=32, d_model=128, d_ffn=257 | random |
| 3 | large | M=128, d_model=256, d_ffn=512 | random |


数据在运行时由固定种子生成，不需要下载数据集。`--seed` 可改变随机数据；非随机边界用例仍保持定义。默认运行所有规模，这些是适合本地迭代的自定义用例，不等同于官方隐藏测试或排行榜规模。可直接在 `main.cpp` 的 `cases` 中增加尺寸。

校验标准：整数精确相等；浮点逐元素 `abs(actual-reference) <= 3e-4 + 3e-4*abs(reference)`，NaN/Inf 一律失败。输入只读区、输出边界保护区也会检查。计时前进行两次独立正确性校验，失败用例显示 FAIL 并跳过性能计时。

## 实现版本

| ID | 名称 | 默认 |
| --- | --- | --- |
| 0 | `starter` | 是 |

`solve()` 接口保持不变；在 `solution.cu` 的 `solve_vN()` 中编写各版本的 host 启动逻辑，并在 `kVersions` 中注册。`kDefaultVersion` 显式选择默认版本。新增版本不需要改 `main.cpp`。

```bash
# 以下命令在仓库根目录执行
python3 scripts/run.py 12 -- --list-versions
python3 scripts/run.py 12 -- --version 0 --size small --check-only
python3 scripts/run.py 12 -- --version all --size small --iterations 20
```

也可以用版本名称选择，例如 `--version starter`。版本选择在预热/计时之前完成；每版使用相同种子生成数据，分别校验和计时。详细添加步骤见[总说明](../../README.md#多版本实现)。

## 编译与运行

以下命令在仓库根目录执行：

```bash
cmake --preset release
cmake --build --preset release --target p12_swiglu_mlp -j 4
./build/release/bin/p12_swiglu_mlp --list-cases
./build/release/bin/p12_swiglu_mlp
./build/release/bin/p12_swiglu_mlp --size small --iterations 20 --warmup 3
./build/release/bin/p12_swiglu_mlp --case 0 --check-only
# 仅验证数据/显存/校验链路，不调用待写 solve，也不代表解题通过：
./build/release/bin/p12_swiglu_mlp --self-test
```

也可用 `python3 scripts/run.py 12 -- --size medium` 自动配置、编译和运行。

初始 kernel 留空，正常运行会 FAIL；完成实现后才会出现 PASS 和 CUDA event 的 mean/min/median/p95，以及主机端 solve+等待完成耗时。显存分配、输入上传、输出复位、CPU 参考和校验不计入 solve 时间。更多计时约定见根 README。

## 练习记录

- [ ] 独立实现并通过所有规模
- [ ] 完成至少一种优化并记录前后耗时
- [ ] 能解释边界处理、同步和访存瓶颈

原题单中的知识点、必做版本与面试追问见 [LeetGPU.md](../../LeetGPU.md)。

## 原题单训练要点

核心知识：

- SiLU
- Gating
- GEMM
- Element-wise Fusion
- 中间张量访存

面试追问：

1. SwiGLU 和普通 ReLU MLP 有什么区别？
2. 哪些操作可以融合？
3. 为什么不一定要把整个 MLP 融合成单 Kernel？
4. 如何判断融合是否划算？
