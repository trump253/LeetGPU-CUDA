# 23. INT8 KV-Cache Attention

选做题，按训练主题归入本阶段。 [LeetGPU](https://leetgpu.com/challenges/int8-kv-cache-attention) · [核对的官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/96_int8_kv_cache_attention/challenge.html) · [总使用说明](../../README.md)

## 任务与接口

FP32 `Q/output[H,D]`；int8 `K_int8/V_int8[H,S,D]`；FP32 `k_scale/v_scale[H,S]`。每个 token/头各有一个 scale，反量化 `K=K_int8*k_scale`、`V=V_int8*v_scale`，再做 scaled decode attention。D 是 8 的倍数。无 zero point，无 mask。

所有指针都是 GPU device pointer，所有张量连续、行主序。只需实现 `solution.cu`；`solution.h` 固定接口，`main.cpp` 提供测试数据、CPU 参考结果与主函数。

```cpp
extern "C" void solve(const float* Q, const int8_t* K_int8, const int8_t* V_int8,
                      const float* k_scale, const float* v_scale, float* output, int num_heads,
                      int seq_len, int head_dim);
```

## 本地测试数据

| 编号 | 规模 | 参数 | 数据 |
| --- | --- | --- | --- |
| 0 | tiny | num_heads=1, seq_len=1, head_dim=8 | random |
| 1 | small | num_heads=3, seq_len=17, head_dim=24 | random |
| 2 | small | num_heads=2, seq_len=129, head_dim=40 | random |
| 3 | medium | num_heads=8, seq_len=1025, head_dim=64 | random |
| 4 | large | num_heads=16, seq_len=8192, head_dim=64 | random |


数据在运行时由固定种子生成，不需要下载数据集。`--seed` 可改变随机数据；非随机边界用例仍保持定义。默认运行所有规模，这些是适合本地迭代的自定义用例，不等同于官方隐藏测试或排行榜规模。可直接在 `main.cpp` 的 `cases` 中增加尺寸。

校验标准：整数精确相等；浮点逐元素 `abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)`，NaN/Inf 一律失败。输入只读区、输出边界保护区也会检查。计时前进行两次独立正确性校验，失败用例显示 FAIL 并跳过性能计时。

## 编译与运行

以下命令在仓库根目录执行：

```bash
cmake --preset release
cmake --build --preset release --target p23_int8_kv_attention -j 4
./build/release/bin/p23_int8_kv_attention --list-cases
./build/release/bin/p23_int8_kv_attention
./build/release/bin/p23_int8_kv_attention --size small --iterations 20 --warmup 3
./build/release/bin/p23_int8_kv_attention --case 0 --check-only
# 仅验证数据/显存/校验链路，不调用待写 solve，也不代表解题通过：
./build/release/bin/p23_int8_kv_attention --self-test
```

也可用 `python3 scripts/run.py 23 -- --size medium` 自动配置、编译和运行。

初始 kernel 留空，正常运行会 FAIL；完成实现后才会出现 PASS 和 CUDA event 的 mean/min/median/p95，以及主机端 solve+等待完成耗时。显存分配、输入上传、输出复位、CPU 参考和校验不计入 solve 时间。更多计时约定见根 README。

## 练习记录

- [ ] 独立实现并通过所有规模
- [ ] 完成至少一种优化并记录前后耗时
- [ ] 能解释边界处理、同步和访存瓶颈

原题单中的知识点、必做版本与面试追问见 [LeetGPU.md](../../LeetGPU.md)。

## 原题单训练要点

核心考点：
INT8 KV Cache、反量化、带宽优化、量化误差。
