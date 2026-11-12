# 07. General Matrix Multiplication (GEMM)

核心题。 [LeetGPU](https://leetgpu.com/challenges/general-matrix-multiplication-gemm) · [核对的官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/22_gemm/challenge.html) · [总使用说明](../../README.md)

## 任务与接口

行主序 FP16 `A[M,K]`、`B[K,N]`、输入输出 `C[M,N]`；计算 `C=alpha*(A@B)+beta*C_initial`，alpha/beta 为 FP32，乘加应以 FP32 或更高精度累加后转换为 FP16。每次调用前恢复 C_initial。

所有指针都是 GPU device pointer，所有张量连续、行主序。只需实现 `solution.cu`；`solution.h` 固定接口，`main.cpp` 提供测试数据、CPU 参考结果与主函数。

```cpp
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta);
```

## 本地测试数据

| 编号 | 规模 | 参数 | 数据 |
| --- | --- | --- | --- |
| 0 | tiny | M=16, N=16, K=16 | random |
| 1 | small | M=17, N=33, K=19 | beta |
| 2 | small | M=32, N=17, K=65 | alpha0 |
| 3 | medium | M=128, N=192, K=257 | beta |
| 4 | large | M=512, N=512, K=512 | beta |


数据在运行时由固定种子生成，不需要下载数据集。`--seed` 可改变随机数据；非随机边界用例仍保持定义。默认运行所有规模，这些是适合本地迭代的自定义用例，不等同于官方隐藏测试或排行榜规模。可直接在 `main.cpp` 的 `cases` 中增加尺寸。

校验标准：整数精确相等；浮点逐元素 `abs(actual-reference) <= 2e-3 + 2e-3*abs(reference)`，NaN/Inf 一律失败。输入只读区、输出边界保护区也会检查。计时前进行两次独立正确性校验，失败用例显示 FAIL 并跳过性能计时。

## 编译与运行

以下命令在仓库根目录执行：

```bash
cmake --preset release
cmake --build --preset release --target p07_gemm -j 4
./build/release/bin/p07_gemm --list-cases
./build/release/bin/p07_gemm
./build/release/bin/p07_gemm --size small --iterations 20 --warmup 3
./build/release/bin/p07_gemm --case 0 --check-only
# 仅验证数据/显存/校验链路，不调用待写 solve，也不代表解题通过：
./build/release/bin/p07_gemm --self-test
```

也可用 `python3 scripts/run.py 07 -- --size medium` 自动配置、编译和运行。

初始 kernel 留空，正常运行会 FAIL；完成实现后才会出现 PASS 和 CUDA event 的 mean/min/median/p95，以及主机端 solve+等待完成耗时。显存分配、输入上传、输出复位、CPU 参考和校验不计入 solve 时间。更多计时约定见根 README。

## 练习记录

- [ ] 独立实现并通过所有规模
- [ ] 完成至少一种优化并记录前后耗时
- [ ] 能解释边界处理、同步和访存瓶颈

原题单中的知识点、必做版本与面试追问见 [LeetGPU.md](../../LeetGPU.md)。

## 原题单训练要点

核心知识：

- Matrix Multiplication
- Shared Memory Tiling
- Data Reuse
- Arithmetic Intensity
- Register Blocking
- Thread Coarsening

必须实现：

1. Naive GEMM
2. Shared Memory Tiled GEMM
3. Register Blocking 优化版本（进阶）

面试追问：

1. 为什么 Tiling 可以提高性能？
2. 如何计算 GEMM 的 FLOPs？
3. GEMM 为什么适合 GPU？
4. 如何估算数据搬运量？
5. 什么是 Arithmetic Intensity？
6. Tensor Core 与 CUDA Core 的区别是什么？
7. 为什么工业级 GEMM 通常需要更复杂的优化？
