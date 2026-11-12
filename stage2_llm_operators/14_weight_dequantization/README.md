# 14. Weight Dequantization

核心题。 [LeetGPU](https://leetgpu.com/challenges/weight-dequantization) · [核对的官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/64_weight_dequantization/challenge.html) · [总使用说明](../../README.md)

## 任务与接口

官方本题的 X、S、Y 都是 **FP32**，没有位打包或 zero point。`X/Y[M,N]`，`S[ceil(M/T),ceil(N/T)]`；`Y[r,c]=X[r,c]*S[r/T,c/T]`，T=TILE_SIZE。

所有指针都是 GPU device pointer，所有张量连续、行主序。只需实现 `solution.cu`；`solution.h` 固定接口，`main.cpp` 提供测试数据、CPU 参考结果与主函数。

```cpp
extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE);
```

## 本地测试数据

| 编号 | 规模 | 参数 | 数据 |
| --- | --- | --- | --- |
| 0 | tiny | M=1, N=1, TILE_SIZE=16 | random |
| 1 | small | M=17, N=33, TILE_SIZE=16 | random |
| 2 | small | M=65, N=31, TILE_SIZE=32 | random |
| 3 | medium | M=513, N=1025, TILE_SIZE=64 | random |
| 4 | large | M=2048, N=4097, TILE_SIZE=128 | random |


数据在运行时由固定种子生成，不需要下载数据集。`--seed` 可改变随机数据；非随机边界用例仍保持定义。默认运行所有规模，这些是适合本地迭代的自定义用例，不等同于官方隐藏测试或排行榜规模。可直接在 `main.cpp` 的 `cases` 中增加尺寸。

校验标准：整数精确相等；浮点逐元素 `abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)`，NaN/Inf 一律失败。输入只读区、输出边界保护区也会检查。计时前进行两次独立正确性校验，失败用例显示 FAIL 并跳过性能计时。

## 实现版本

| ID | 名称 | 默认 |
| --- | --- | --- |
| 0 | `starter` | 是 |

`solve()` 接口保持不变；在 `solution.cu` 的 `solve_vN()` 中编写各版本的 host 启动逻辑，并在 `kVersions` 中注册。`kDefaultVersion` 显式选择默认版本。新增版本不需要改 `main.cpp`。

```bash
# 以下命令在仓库根目录执行
python3 scripts/run.py 14 -- --list-versions
python3 scripts/run.py 14 -- --version 0 --size small --check-only
python3 scripts/run.py 14 -- --version all --size small --iterations 20
```

也可以用版本名称选择，例如 `--version starter`。版本选择在预热/计时之前完成；每版使用相同种子生成数据，分别校验和计时。详细添加步骤见[总说明](../../README.md#多版本实现)。

## 编译与运行

以下命令在仓库根目录执行：

```bash
cmake --preset release
cmake --build --preset release --target p14_weight_dequantization -j 4
./build/release/bin/p14_weight_dequantization --list-cases
./build/release/bin/p14_weight_dequantization
./build/release/bin/p14_weight_dequantization --size small --iterations 20 --warmup 3
./build/release/bin/p14_weight_dequantization --case 0 --check-only
# 仅验证数据/显存/校验链路，不调用待写 solve，也不代表解题通过：
./build/release/bin/p14_weight_dequantization --self-test
```

也可用 `python3 scripts/run.py 14 -- --size medium` 自动配置、编译和运行。

初始 kernel 留空，正常运行会 FAIL；完成实现后才会出现 PASS 和 CUDA event 的 mean/min/median/p95，以及主机端 solve+等待完成耗时。显存分配、输入上传、输出复位、CPU 参考和校验不计入 solve 时间。更多计时约定见根 README。

## 练习记录

- [ ] 独立实现并通过所有规模
- [ ] 完成至少一种优化并记录前后耗时
- [ ] 能解释边界处理、同步和访存瓶颈

原题单中的知识点、必做版本与面试追问见 [LeetGPU.md](../../LeetGPU.md)。

## 原题单训练要点

核心知识：

- Quantization
- Dequantization
- Scale / Zero Point
- Bit Packing
- Memory Bandwidth

面试追问：

1. 为什么要进行权重量化？
2. Per-channel 与 Per-group 量化的区别？
3. Dequantization 为什么可能成为瓶颈？
4. 如何减少反量化中间张量？
