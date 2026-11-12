# 02. Matrix Transpose

核心题。 [LeetGPU](https://leetgpu.com/challenges/matrix-transpose) · [核对的官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/3_matrix_transpose/challenge.html) · [总使用说明](../../README.md)

## 任务与接口

将 FP32 行主序 `input[rows,cols]` 转置为 `output[cols,rows]`：`output[c*rows+r]=input[r*cols+c]`。

所有指针都是 GPU device pointer，所有张量连续、行主序。只需实现 `solution.cu`；`solution.h` 固定接口，`main.cpp` 提供测试数据、CPU 参考结果与主函数。

```cpp
extern "C" void solve(const float* input, float* output, int rows, int cols);
```

## 本地测试数据

| 编号 | 规模 | 参数 | 数据 |
| --- | --- | --- | --- |
| 0 | tiny | rows=1, cols=1 | random |
| 1 | small | rows=1, cols=37 | random |
| 2 | small | rows=37, cols=1 | random |
| 3 | small | rows=31, cols=33 | random |
| 4 | medium | rows=257, cols=509 | random |
| 5 | large | rows=2048, cols=4097 | random |


数据在运行时由固定种子生成，不需要下载数据集。`--seed` 可改变随机数据；非随机边界用例仍保持定义。默认运行所有规模，这些是适合本地迭代的自定义用例，不等同于官方隐藏测试或排行榜规模。可直接在 `main.cpp` 的 `cases` 中增加尺寸。

校验标准：整数精确相等；浮点逐元素 `abs(actual-reference) <= 0 + 0*abs(reference)`，NaN/Inf 一律失败。输入只读区、输出边界保护区也会检查。计时前进行两次独立正确性校验，失败用例显示 FAIL 并跳过性能计时。

## 编译与运行

以下命令在仓库根目录执行：

```bash
cmake --preset release
cmake --build --preset release --target p02_matrix_transpose -j 4
./build/release/bin/p02_matrix_transpose --list-cases
./build/release/bin/p02_matrix_transpose
./build/release/bin/p02_matrix_transpose --size small --iterations 20 --warmup 3
./build/release/bin/p02_matrix_transpose --case 0 --check-only
# 仅验证数据/显存/校验链路，不调用待写 solve，也不代表解题通过：
./build/release/bin/p02_matrix_transpose --self-test
```

也可用 `python3 scripts/run.py 02 -- --size medium` 自动配置、编译和运行。

初始 kernel 留空，正常运行会 FAIL；完成实现后才会出现 PASS 和 CUDA event 的 mean/min/median/p95，以及主机端 solve+等待完成耗时。显存分配、输入上传、输出复位、CPU 参考和校验不计入 solve 时间。更多计时约定见根 README。

## 练习记录

- [ ] 独立实现并通过所有规模
- [ ] 完成至少一种优化并记录前后耗时
- [ ] 能解释边界处理、同步和访存瓶颈

原题单中的知识点、必做版本与面试追问见 [LeetGPU.md](../../LeetGPU.md)。

## 原题单训练要点

核心知识：

- Row-major 地址映射
- Global Memory Coalescing
- Shared Memory Tiling
- Bank Conflict
- Shared Memory Padding

必须实现：

1. Naive Transpose
2. Shared Memory Tiled Transpose
3. Padding 优化版本

面试追问：

1. 为什么直接转置可能造成非合并访存？
2. 为什么 Shared Memory 可以提高性能？
3. 为什么 tile[32][33] 常被使用？
4. Bank Conflict 是什么？
