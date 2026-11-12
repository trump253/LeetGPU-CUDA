# 05. Prefix Sum

核心题。 [LeetGPU](https://leetgpu.com/challenges/prefix-sum) · [核对的官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/16_prefix_sum/challenge.html) · [总使用说明](../../README.md)

## 任务与接口

计算 **inclusive scan**：`output[i]=sum(input[0..i])`，FP32 `[N]`。包含负数及跨 block 尾部。

所有指针都是 GPU device pointer，所有张量连续、行主序。只需实现 `solution.cu`；`solution.h` 固定接口，`main.cpp` 提供测试数据、CPU 参考结果与主函数。

```cpp
extern "C" void solve(const float* input, float* output, int N);
```

## 本地测试数据

| 编号 | 规模 | 参数 | 数据 |
| --- | --- | --- | --- |
| 0 | tiny | N=1 | random |
| 1 | small | N=31 | signed |
| 2 | small | N=257 | zeros |
| 3 | small | N=1003 | random |
| 4 | medium | N=65539 | random |
| 5 | large | N=4194307 | random |


数据在运行时由固定种子生成，不需要下载数据集。`--seed` 可改变随机数据；非随机边界用例仍保持定义。默认运行所有规模，这些是适合本地迭代的自定义用例，不等同于官方隐藏测试或排行榜规模。可直接在 `main.cpp` 的 `cases` 中增加尺寸。

校验标准：整数精确相等；浮点逐元素 `abs(actual-reference) <= 2e-3 + 2e-4*abs(reference)`，NaN/Inf 一律失败。输入只读区、输出边界保护区也会检查。计时前进行两次独立正确性校验，失败用例显示 FAIL 并跳过性能计时。

## 编译与运行

以下命令在仓库根目录执行：

```bash
cmake --preset release
cmake --build --preset release --target p05_prefix_sum -j 4
./build/release/bin/p05_prefix_sum --list-cases
./build/release/bin/p05_prefix_sum
./build/release/bin/p05_prefix_sum --size small --iterations 20 --warmup 3
./build/release/bin/p05_prefix_sum --case 0 --check-only
# 仅验证数据/显存/校验链路，不调用待写 solve，也不代表解题通过：
./build/release/bin/p05_prefix_sum --self-test
```

也可用 `python3 scripts/run.py 05 -- --size medium` 自动配置、编译和运行。

初始 kernel 留空，正常运行会 FAIL；完成实现后才会出现 PASS 和 CUDA event 的 mean/min/median/p95，以及主机端 solve+等待完成耗时。显存分配、输入上传、输出复位、CPU 参考和校验不计入 solve 时间。更多计时约定见根 README。

## 练习记录

- [ ] 独立实现并通过所有规模
- [ ] 完成至少一种优化并记录前后耗时
- [ ] 能解释边界处理、同步和访存瓶颈

原题单中的知识点、必做版本与面试追问见 [LeetGPU.md](../../LeetGPU.md)。

## 原题单训练要点

核心知识：

- Parallel Scan
- Inclusive / Exclusive Scan
- Shared Memory
- Block 间结果合并
- Blelloch Scan

面试追问：

1. Scan 和 Reduction 有什么区别？
2. 为什么简单的串行 Scan 不适合 GPU？
3. 如何跨 Block 完成 Scan？
4. Work-efficient Scan 是什么意思？
