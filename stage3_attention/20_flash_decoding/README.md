# 20. Flash Decoding Attention

核心题。 [LeetGPU](https://leetgpu.com/challenges/flash-decoding-attention) · [核对的官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/hard/141_flash_decoding_attention/challenge.html) · [总使用说明](../../README.md)

## 任务与接口

FP32 `Q/output[T,H,D]`、`K/V[S,H,D]`，注意序列维在 head 维之前。每个 query 对全部 S 个 KV token 做无 mask attention，缩放系数 `1/sqrt(D)`。T 通常远小于 S。包含长序列、非整齐尾部以及较大 score。

所有指针都是 GPU device pointer，所有张量连续、行主序。只需实现 `solution.cu`；`solution.h` 固定接口，`main.cpp` 提供测试数据、CPU 参考结果与主函数。

```cpp
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int T, int S,
                      int H, int D);
```

## 本地测试数据

| 编号 | 规模 | 参数 | 数据 |
| --- | --- | --- | --- |
| 0 | tiny | T=1, S=1, H=1, D=1 | random |
| 1 | small | T=2, S=33, H=3, D=7 | random |
| 2 | small | T=1, S=129, H=2, D=32 | large_scores |
| 3 | medium | T=2, S=4097, H=8, D=64 | random |
| 4 | large | T=1, S=32768, H=16, D=64 | random |


数据在运行时由固定种子生成，不需要下载数据集。`--seed` 可改变随机数据；非随机边界用例仍保持定义。默认运行所有规模，这些是适合本地迭代的自定义用例，不等同于官方隐藏测试或排行榜规模。可直接在 `main.cpp` 的 `cases` 中增加尺寸。

校验标准：整数精确相等；浮点逐元素 `abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)`，NaN/Inf 一律失败。输入只读区、输出边界保护区也会检查。计时前进行两次独立正确性校验，失败用例显示 FAIL 并跳过性能计时。

## 编译与运行

以下命令在仓库根目录执行：

```bash
cmake --preset release
cmake --build --preset release --target p20_flash_decoding -j 4
./build/release/bin/p20_flash_decoding --list-cases
./build/release/bin/p20_flash_decoding
./build/release/bin/p20_flash_decoding --size small --iterations 20 --warmup 3
./build/release/bin/p20_flash_decoding --case 0 --check-only
# 仅验证数据/显存/校验链路，不调用待写 solve，也不代表解题通过：
./build/release/bin/p20_flash_decoding --self-test
```

也可用 `python3 scripts/run.py 20 -- --size medium` 自动配置、编译和运行。

初始 kernel 留空，正常运行会 FAIL；完成实现后才会出现 PASS 和 CUDA event 的 mean/min/median/p95，以及主机端 solve+等待完成耗时。显存分配、输入上传、输出复位、CPU 参考和校验不计入 solve 时间。更多计时约定见根 README。

## 练习记录

- [ ] 独立实现并通过所有规模
- [ ] 完成至少一种优化并记录前后耗时
- [ ] 能解释边界处理、同步和访存瓶颈

原题单中的知识点、必做版本与面试追问见 [LeetGPU.md](../../LeetGPU.md)。

## 原题单训练要点

核心知识：

- Flash Decoding
- Split-KV
- Online Softmax
- Parallel Reduction
- 长上下文优化
- Occupancy

面试追问：

1. FlashAttention 和 Flash Decoding 有什么区别？
2. 为什么 Decode 阶段经常面临并行度不足？
3. Split-KV 什么时候值得使用？
4. Split 越多是否越好？
5. 如何处理多个 Split 之间的数值稳定性？
