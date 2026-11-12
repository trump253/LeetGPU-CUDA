# 08. Softmax

核心题。 [LeetGPU](https://leetgpu.com/challenges/softmax) · [核对的官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/5_softmax/challenge.html) · [总使用说明](../../README.md)

## 任务与接口

对整个 FP32 向量 `[N]` 计算稳定 softmax：`exp(x-max(x))/sum(exp(x-max(x)))`。不是逐行二维 softmax。测试包含全零及 ±80 的大幅值。

所有指针都是 GPU device pointer，所有张量连续、行主序。只需实现 `solution.cu`；`solution.h` 固定接口，`main.cpp` 提供测试数据、CPU 参考结果与主函数。

```cpp
extern "C" void solve(const float* input, float* output, int N);
```

## 本地测试数据

| 编号 | 规模 | 参数 | 数据 |
| --- | --- | --- | --- |
| 0 | tiny | N=1 | random |
| 1 | small | N=31 | zeros |
| 2 | small | N=257 | extreme |
| 3 | small | N=1003 | random |
| 4 | medium | N=65539 | random |
| 5 | large | N=500000 | extreme |


数据在运行时由固定种子生成，不需要下载数据集。`--seed` 可改变随机数据；非随机边界用例仍保持定义。默认运行所有规模，这些是适合本地迭代的自定义用例，不等同于官方隐藏测试或排行榜规模。可直接在 `main.cpp` 的 `cases` 中增加尺寸。

校验标准：整数精确相等；浮点逐元素 `abs(actual-reference) <= 1e-7 + 2e-4*abs(reference)`，NaN/Inf 一律失败。输入只读区、输出边界保护区也会检查。计时前进行两次独立正确性校验，失败用例显示 FAIL 并跳过性能计时。

## 实现版本

| ID | 名称 | 默认 |
| --- | --- | --- |
| 0 | `starter` | 是 |

`solve()` 接口保持不变；在 `solution.cu` 的 `solve_vN()` 中编写各版本的 host 启动逻辑，并在 `kVersions` 中注册。`kDefaultVersion` 显式选择默认版本。新增版本不需要改 `main.cpp`。

```bash
# 以下命令在仓库根目录执行
python3 scripts/run.py 08 -- --list-versions
python3 scripts/run.py 08 -- --version 0 --size small --check-only
python3 scripts/run.py 08 -- --version all --size small --iterations 20
```

也可以用版本名称选择，例如 `--version starter`。版本选择在预热/计时之前完成；每版使用相同种子生成数据，分别校验和计时。详细添加步骤见[总说明](../../README.md#多版本实现)。

## 编译与运行

以下命令在仓库根目录执行：

```bash
cmake --preset release
cmake --build --preset release --target p08_softmax -j 4
./build/release/bin/p08_softmax --list-cases
./build/release/bin/p08_softmax
./build/release/bin/p08_softmax --size small --iterations 20 --warmup 3
./build/release/bin/p08_softmax --case 0 --check-only
# 仅验证数据/显存/校验链路，不调用待写 solve，也不代表解题通过：
./build/release/bin/p08_softmax --self-test
```

也可用 `python3 scripts/run.py 08 -- --size medium` 自动配置、编译和运行。

初始 kernel 留空，正常运行会 FAIL；完成实现后才会出现 PASS 和 CUDA event 的 mean/min/median/p95，以及主机端 solve+等待完成耗时。显存分配、输入上传、输出复位、CPU 参考和校验不计入 solve 时间。更多计时约定见根 README。

## 练习记录

- [ ] 独立实现并通过所有规模
- [ ] 完成至少一种优化并记录前后耗时
- [ ] 能解释边界处理、同步和访存瓶颈

原题单中的知识点、必做版本与面试追问见 [LeetGPU.md](../../LeetGPU.md)。

## 原题单训练要点

核心知识：

- Max Reduction
- Sum Reduction
- Numerical Stability
- Warp Shuffle
- Online Softmax

必须实现：

1. 基础 Softmax
2. 数值稳定版本
3. Warp / Block 并行优化

面试追问：

1. 为什么 Softmax 要减去最大值？
2. Softmax 为什么需要两次 Reduction？
3. 如何减少 Global Memory 访问？
4. Online Softmax 是什么？
5. FlashAttention 如何利用 Online Softmax？
