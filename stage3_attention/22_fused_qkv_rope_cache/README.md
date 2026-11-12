# 22. Fused QKV Projection with RoPE and KV Cache Update

选做题，按训练主题归入本阶段。 [LeetGPU](https://leetgpu.com/challenges/fused-qkv-projection-with-rope-and-kv-cache-update) · [核对的官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/115_fused_qkv_rope_kv_cache_update/challenge.html) · [总使用说明](../../README.md)

## 任务与接口

FP32 `x[B,d_model]`、`W_qkv[d_model,(Hq+2*Hkv)*D]`、`cos_sin_cache[Smax,D]`、`K_cache/V_cache[B,Hkv,Smax,D]`、`Q_out[B,Hq,D]`，int32 `positions[B]`。先做 QKV 投影，依次切分 Q/K/V；对 Q、K 做 half-split RoPE，表的前 D/2 是 cos、后 D/2 是 sin。将 K/V 写到各 batch 的 positions[b]，其他 cache 槽必须原样保留。该接口是连续 cache，没有 block_table。

所有指针都是 GPU device pointer，所有张量连续、行主序。只需实现 `solution.cu`；`solution.h` 固定接口，`main.cpp` 提供测试数据、CPU 参考结果与主函数。

```cpp
extern "C" void solve(const float* x, const float* W_qkv, const float* cos_sin_cache,
                      const int* positions, float* K_cache, float* V_cache, float* Q_out, int B,
                      int d_model, int H_q, int H_kv, int D, int S_max);
```

## 本地测试数据

| 编号 | 规模 | 参数 | 数据 |
| --- | --- | --- | --- |
| 0 | tiny | B=1, d_model=2, H_q=1, H_kv=1, D=2, S_max=2 | random |
| 1 | small | B=3, d_model=17, H_q=4, H_kv=2, D=6, S_max=7 | random |
| 2 | medium | B=8, d_model=128, H_q=8, H_kv=2, D=32, S_max=64 | random |
| 3 | large | B=16, d_model=512, H_q=8, H_kv=2, D=64, S_max=128 | random |


数据在运行时由固定种子生成，不需要下载数据集。`--seed` 可改变随机数据；非随机边界用例仍保持定义。默认运行所有规模，这些是适合本地迭代的自定义用例，不等同于官方隐藏测试或排行榜规模。可直接在 `main.cpp` 的 `cases` 中增加尺寸。

校验标准：整数精确相等；浮点逐元素 `abs(actual-reference) <= 3e-4 + 3e-4*abs(reference)`，NaN/Inf 一律失败。输入只读区、输出边界保护区也会检查。计时前进行两次独立正确性校验，失败用例显示 FAIL 并跳过性能计时。

## 编译与运行

以下命令在仓库根目录执行：

```bash
cmake --preset release
cmake --build --preset release --target p22_fused_qkv_rope_cache -j 4
./build/release/bin/p22_fused_qkv_rope_cache --list-cases
./build/release/bin/p22_fused_qkv_rope_cache
./build/release/bin/p22_fused_qkv_rope_cache --size small --iterations 20 --warmup 3
./build/release/bin/p22_fused_qkv_rope_cache --case 0 --check-only
# 仅验证数据/显存/校验链路，不调用待写 solve，也不代表解题通过：
./build/release/bin/p22_fused_qkv_rope_cache --self-test
```

也可用 `python3 scripts/run.py 22 -- --size medium` 自动配置、编译和运行。

初始 kernel 留空，正常运行会 FAIL；完成实现后才会出现 PASS 和 CUDA event 的 mean/min/median/p95，以及主机端 solve+等待完成耗时。显存分配、输入上传、输出复位、CPU 参考和校验不计入 solve 时间。更多计时约定见根 README。

## 练习记录

- [ ] 独立实现并通过所有规模
- [ ] 完成至少一种优化并记录前后耗时
- [ ] 能解释边界处理、同步和访存瓶颈

原题单中的知识点、必做版本与面试追问见 [LeetGPU.md](../../LeetGPU.md)。

## 原题单训练要点

核心考点：
QKV Projection、RoPE、KV Cache 写入、Kernel Fusion。
