# 25. Top K Selection

选做题，按训练主题归入本阶段。 [LeetGPU](https://leetgpu.com/challenges/top-k-selection) · [核对的官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/29_top_k_selection/challenge.html) · [总使用说明](../../README.md)

## 任务与接口

FP32 输入 `[N]`，将最大的 k 个值按非递增顺序写入 FP32 `[k]`，无需输出索引。包含 k=1、k=N、重复值和负数。

所有指针都是 GPU device pointer，所有张量连续、行主序。只需实现 `solution.cu`；`solution.h` 固定接口，`main.cpp` 提供测试数据、CPU 参考结果与主函数。

```cpp
extern "C" void solve(const float* input, float* output, int N, int k);
```

## 本地测试数据

| 编号 | 规模 | 参数 | 数据 |
| --- | --- | --- | --- |
| 0 | tiny | N=1, k=1 | random |
| 1 | small | N=31, k=1 | random |
| 2 | small | N=257, k=257 | duplicates |
| 3 | medium | N=65539, k=37 | random |
| 4 | large | N=4194307, k=100 | duplicates |


数据在运行时由固定种子生成，不需要下载数据集。`--seed` 可改变随机数据；非随机边界用例仍保持定义。默认运行所有规模，这些是适合本地迭代的自定义用例，不等同于官方隐藏测试或排行榜规模。可直接在 `main.cpp` 的 `cases` 中增加尺寸。

校验标准：整数精确相等；浮点逐元素 `abs(actual-reference) <= 0 + 0*abs(reference)`，NaN/Inf 一律失败。输入只读区、输出边界保护区也会检查。计时前进行两次独立正确性校验，失败用例显示 FAIL 并跳过性能计时。

## 编译与运行

以下命令在仓库根目录执行：

```bash
cmake --preset release
cmake --build --preset release --target p25_top_k_selection -j 4
./build/release/bin/p25_top_k_selection --list-cases
./build/release/bin/p25_top_k_selection
./build/release/bin/p25_top_k_selection --size small --iterations 20 --warmup 3
./build/release/bin/p25_top_k_selection --case 0 --check-only
# 仅验证数据/显存/校验链路，不调用待写 solve，也不代表解题通过：
./build/release/bin/p25_top_k_selection --self-test
```

也可用 `python3 scripts/run.py 25 -- --size medium` 自动配置、编译和运行。

初始 kernel 留空，正常运行会 FAIL；完成实现后才会出现 PASS 和 CUDA event 的 mean/min/median/p95，以及主机端 solve+等待完成耗时。显存分配、输入上传、输出复位、CPU 参考和校验不计入 solve 时间。更多计时约定见根 README。

## 练习记录

- [ ] 独立实现并通过所有规模
- [ ] 完成至少一种优化并记录前后耗时
- [ ] 能解释边界处理、同步和访存瓶颈

原题单中的知识点、必做版本与面试追问见 [LeetGPU.md](../../LeetGPU.md)。

## 原题单训练要点

核心考点：
Parallel Selection、Warp 协作、Sorting、Top-K。
