# 17. Grouped Query Attention (GQA)

核心题。 [LeetGPU](https://leetgpu.com/challenges/grouped-query-attention) · [核对的官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/80_grouped_query_attention/challenge.html) · [总使用说明](../../README.md)

## 任务与接口

FP32 `Q/output[Hq,S,D]`、`K/V[Hkv,S,D]`，是完整序列 attention，非单 token decode。Hq 必须是 Hkv 的整数倍，D 是 8 的倍数。query head h 使用 KV head `h/(Hq/Hkv)`；每个头计算无 mask 的 scaled dot-product attention。

所有指针都是 GPU device pointer，所有张量连续、行主序。只需实现 `solution.cu`；`solution.h` 固定接口，`main.cpp` 提供测试数据、CPU 参考结果与主函数。

```cpp
extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int num_q_heads, int num_kv_heads, int seq_len, int head_dim);
```

## 本地测试数据

| 编号 | 规模 | 参数 | 数据 |
| --- | --- | --- | --- |
| 0 | tiny | num_q_heads=1, num_kv_heads=1, seq_len=1, head_dim=8 | random |
| 1 | small | num_q_heads=4, num_kv_heads=1, seq_len=17, head_dim=24 | random |
| 2 | small | num_q_heads=6, num_kv_heads=2, seq_len=33, head_dim=40 | random |
| 3 | medium | num_q_heads=8, num_kv_heads=2, seq_len=128, head_dim=64 | random |
| 4 | large | num_q_heads=8, num_kv_heads=2, seq_len=512, head_dim=64 | random |


数据在运行时由固定种子生成，不需要下载数据集。`--seed` 可改变随机数据；非随机边界用例仍保持定义。默认运行所有规模，这些是适合本地迭代的自定义用例，不等同于官方隐藏测试或排行榜规模。可直接在 `main.cpp` 的 `cases` 中增加尺寸。

校验标准：整数精确相等；浮点逐元素 `abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)`，NaN/Inf 一律失败。输入只读区、输出边界保护区也会检查。计时前进行两次独立正确性校验，失败用例显示 FAIL 并跳过性能计时。

## 实现版本

| ID | 名称 | 默认 |
| --- | --- | --- |
| 0 | `starter` | 是 |

`solve()` 接口保持不变；在 `solution.cu` 的 `solve_vN()` 中编写各版本的 host 启动逻辑，并在 `kVersions` 中注册。`kDefaultVersion` 显式选择默认版本。新增版本不需要改 `main.cpp`。

```bash
# 以下命令在仓库根目录执行
python3 scripts/run.py 17 -- --list-versions
python3 scripts/run.py 17 -- --version 0 --size small --check-only
python3 scripts/run.py 17 -- --version all --size small --iterations 20
```

也可以用版本名称选择，例如 `--version starter`。版本选择在预热/计时之前完成；每版使用相同种子生成数据，分别校验和计时。详细添加步骤见[总说明](../../README.md#多版本实现)。

## 编译与运行

以下命令在仓库根目录执行：

```bash
cmake --preset release
cmake --build --preset release --target p17_grouped_query_attention -j 4
./build/release/bin/p17_grouped_query_attention --list-cases
./build/release/bin/p17_grouped_query_attention
./build/release/bin/p17_grouped_query_attention --size small --iterations 20 --warmup 3
./build/release/bin/p17_grouped_query_attention --case 0 --check-only
# 仅验证数据/显存/校验链路，不调用待写 solve，也不代表解题通过：
./build/release/bin/p17_grouped_query_attention --self-test
```

也可用 `python3 scripts/run.py 17 -- --size medium` 自动配置、编译和运行。

初始 kernel 留空，正常运行会 FAIL；完成实现后才会出现 PASS 和 CUDA event 的 mean/min/median/p95，以及主机端 solve+等待完成耗时。显存分配、输入上传、输出复位、CPU 参考和校验不计入 solve 时间。更多计时约定见根 README。

## 练习记录

- [ ] 独立实现并通过所有规模
- [ ] 完成至少一种优化并记录前后耗时
- [ ] 能解释边界处理、同步和访存瓶颈

原题单中的知识点、必做版本与面试追问见 [LeetGPU.md](../../LeetGPU.md)。

## 原题单训练要点

核心知识：

- MHA / MQA / GQA
- Query Head 映射
- KV Head 复用
- Attention Decode
- KV Cache 带宽

面试追问：

1. MHA / MQA / GQA 有什么区别？
2. GQA 为什么能降低 KV Cache 大小？
3. Q Head 如何映射到 KV Head？
4. GQA 如何影响 Memory Bandwidth？
5. GQA 是否总能提高推理性能？
