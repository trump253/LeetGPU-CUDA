# 19. Split-KV Attention Reduction

核心题。 [LeetGPU](https://leetgpu.com/challenges/split-kv-attention-reduction) · [核对的官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/128_split_kv_attention_reduction/challenge.html) · [总使用说明](../../README.md)

## 任务与接口

FP32 `partial_out[S,H,D]`、`partial_lse[S,H]`，输出 `[H,D]`。对每个头，将 split 输出按 `softmax(partial_lse[:,h])` 加权合并。LSE 使用自然对数，测试包括相等 LSE 和 ±80 的差异，不能直接平均 partial_out。

所有指针都是 GPU device pointer，所有张量连续、行主序。只需实现 `solution.cu`；`solution.h` 固定接口，`main.cpp` 提供测试数据、CPU 参考结果与主函数。

```cpp
extern "C" void solve(const float* partial_out, const float* partial_lse, float* output,
                      int num_splits, int num_heads, int head_dim);
```

## 本地测试数据

| 编号 | 规模 | 参数 | 数据 |
| --- | --- | --- | --- |
| 0 | tiny | num_splits=1, num_heads=1, head_dim=1 | random |
| 1 | small | num_splits=3, num_heads=2, head_dim=7 | equal |
| 2 | small | num_splits=17, num_heads=3, head_dim=33 | extreme |
| 3 | medium | num_splits=64, num_heads=32, head_dim=64 | random |
| 4 | large | num_splits=128, num_heads=64, head_dim=128 | extreme |


数据在运行时由固定种子生成，不需要下载数据集。`--seed` 可改变随机数据；非随机边界用例仍保持定义。默认运行所有规模，这些是适合本地迭代的自定义用例，不等同于官方隐藏测试或排行榜规模。可直接在 `main.cpp` 的 `cases` 中增加尺寸。

校验标准：整数精确相等；浮点逐元素 `abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)`，NaN/Inf 一律失败。输入只读区、输出边界保护区也会检查。计时前进行两次独立正确性校验，失败用例显示 FAIL 并跳过性能计时。

## 编译与运行

以下命令在仓库根目录执行：

```bash
cmake --preset release
cmake --build --preset release --target p19_split_kv_reduction -j 4
./build/release/bin/p19_split_kv_reduction --list-cases
./build/release/bin/p19_split_kv_reduction
./build/release/bin/p19_split_kv_reduction --size small --iterations 20 --warmup 3
./build/release/bin/p19_split_kv_reduction --case 0 --check-only
# 仅验证数据/显存/校验链路，不调用待写 solve，也不代表解题通过：
./build/release/bin/p19_split_kv_reduction --self-test
```

也可用 `python3 scripts/run.py 19 -- --size medium` 自动配置、编译和运行。

初始 kernel 留空，正常运行会 FAIL；完成实现后才会出现 PASS 和 CUDA event 的 mean/min/median/p95，以及主机端 solve+等待完成耗时。显存分配、输入上传、输出复位、CPU 参考和校验不计入 solve 时间。更多计时约定见根 README。

## 练习记录

- [ ] 独立实现并通过所有规模
- [ ] 完成至少一种优化并记录前后耗时
- [ ] 能解释边界处理、同步和访存瓶颈

原题单中的知识点、必做版本与面试追问见 [LeetGPU.md](../../LeetGPU.md)。

## 原题单训练要点

核心知识：

- Split-KV
- LogSumExp
- Online Softmax
- Partial Attention Output
- 多 CTA 结果合并

面试追问：

1. 为什么长上下文 Decode 要 Split-KV？
2. 不同 Split 的 Softmax 结果为什么不能直接相加？
3. 如何使用 LSE 合并不同 Split？
4. Split-KV 为什么可以提高 SM 利用率？
