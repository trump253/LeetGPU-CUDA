# 21. MoE Top-K Gating

核心题。 [LeetGPU](https://leetgpu.com/challenges/moe-top-k-gating) · [核对的官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/67_moe_topk_gating/challenge.html) · [总使用说明](../../README.md)

## 任务与接口

FP32 `logits[M,E]`，输出 FP32 `topk_weights[M,k]`、int32 `topk_indices[M,k]`。每行选最大 k 个 logit，按降序输出索引，仅对这 k 个值做 softmax。为避免平台间 tie-breaking 差异，本地 MoE 输入每行无重复 logit；普通 Top-K 第 25 题覆盖重复值。

所有指针都是 GPU device pointer，所有张量连续、行主序。只需实现 `solution.cu`；`solution.h` 固定接口，`main.cpp` 提供测试数据、CPU 参考结果与主函数。

```cpp
extern "C" void solve(const float* logits, float* topk_weights, int* topk_indices, int M, int E,
                      int k);
```

## 本地测试数据

| 编号 | 规模 | 参数 | 数据 |
| --- | --- | --- | --- |
| 0 | tiny | M=1, E=1, k=1 | random |
| 1 | small | M=3, E=17, k=1 | random |
| 2 | small | M=7, E=31, k=31 | random |
| 3 | medium | M=128, E=64, k=2 | random |
| 4 | large | M=4096, E=256, k=8 | random |


数据在运行时由固定种子生成，不需要下载数据集。`--seed` 可改变随机数据；非随机边界用例仍保持定义。默认运行所有规模，这些是适合本地迭代的自定义用例，不等同于官方隐藏测试或排行榜规模。可直接在 `main.cpp` 的 `cases` 中增加尺寸。

校验标准：整数精确相等；浮点逐元素 `abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)`，NaN/Inf 一律失败。输入只读区、输出边界保护区也会检查。计时前进行两次独立正确性校验，失败用例显示 FAIL 并跳过性能计时。

## 编译与运行

以下命令在仓库根目录执行：

```bash
cmake --preset release
cmake --build --preset release --target p21_moe_topk_gating -j 4
./build/release/bin/p21_moe_topk_gating --list-cases
./build/release/bin/p21_moe_topk_gating
./build/release/bin/p21_moe_topk_gating --size small --iterations 20 --warmup 3
./build/release/bin/p21_moe_topk_gating --case 0 --check-only
# 仅验证数据/显存/校验链路，不调用待写 solve，也不代表解题通过：
./build/release/bin/p21_moe_topk_gating --self-test
```

也可用 `python3 scripts/run.py 21 -- --size medium` 自动配置、编译和运行。

初始 kernel 留空，正常运行会 FAIL；完成实现后才会出现 PASS 和 CUDA event 的 mean/min/median/p95，以及主机端 solve+等待完成耗时。显存分配、输入上传、输出复位、CPU 参考和校验不计入 solve 时间。更多计时约定见根 README。

## 练习记录

- [ ] 独立实现并通过所有规模
- [ ] 完成至少一种优化并记录前后耗时
- [ ] 能解释边界处理、同步和访存瓶颈

原题单中的知识点、必做版本与面试追问见 [LeetGPU.md](../../LeetGPU.md)。

## 原题单训练要点

核心知识：

- Top-K
- Expert Routing
- Sorting / Selection
- Token Dispatch
- MoE Inference

面试追问：

1. MoE 为什么需要 Top-K Routing？
2. Top-K 如何在 GPU 上实现？
3. Expert Routing 有什么性能瓶颈？
4. MoE 和 Dense Transformer 的计算特点有何区别？
