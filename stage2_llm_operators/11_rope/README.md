# 11. Rotary Positional Embedding (RoPE)

核心题。 [LeetGPU](https://leetgpu.com/challenges/rotary-positional-embedding) · [核对的官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/61_rope_embedding/challenge.html) · [总使用说明](../../README.md)

## 任务与接口

FP32 `Q/cos/sin/output[M,D]`，D 为偶数。采用 **half-split / NeoX** 配对：`rotate_half(Q)=[-Q_second_half,Q_first_half]`，结果为 `Q*cos+rotate_half(Q)*sin`。cos/sin 的前后半维重复，位置已编码在表中。

所有指针都是 GPU device pointer，所有张量连续、行主序。只需实现 `solution.cu`；`solution.h` 固定接口，`main.cpp` 提供测试数据、CPU 参考结果与主函数。

```cpp
extern "C" void solve(float* Q, float* cos, float* sin, float* output, int M, int D);
```

## 本地测试数据

| 编号 | 规模 | 参数 | 数据 |
| --- | --- | --- | --- |
| 0 | tiny | M=1, D=2 | random |
| 1 | small | M=3, D=6 | random |
| 2 | small | M=17, D=130 | random |
| 3 | medium | M=1024, D=128 | random |
| 4 | large | M=8192, D=128 | random |


数据在运行时由固定种子生成，不需要下载数据集。`--seed` 可改变随机数据；非随机边界用例仍保持定义。默认运行所有规模，这些是适合本地迭代的自定义用例，不等同于官方隐藏测试或排行榜规模。可直接在 `main.cpp` 的 `cases` 中增加尺寸。

校验标准：整数精确相等；浮点逐元素 `abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)`，NaN/Inf 一律失败。输入只读区、输出边界保护区也会检查。计时前进行两次独立正确性校验，失败用例显示 FAIL 并跳过性能计时。

## 编译与运行

以下命令在仓库根目录执行：

```bash
cmake --preset release
cmake --build --preset release --target p11_rope -j 4
./build/release/bin/p11_rope --list-cases
./build/release/bin/p11_rope
./build/release/bin/p11_rope --size small --iterations 20 --warmup 3
./build/release/bin/p11_rope --case 0 --check-only
# 仅验证数据/显存/校验链路，不调用待写 solve，也不代表解题通过：
./build/release/bin/p11_rope --self-test
```

也可用 `python3 scripts/run.py 11 -- --size medium` 自动配置、编译和运行。

初始 kernel 留空，正常运行会 FAIL；完成实现后才会出现 PASS 和 CUDA event 的 mean/min/median/p95，以及主机端 solve+等待完成耗时。显存分配、输入上传、输出复位、CPU 参考和校验不计入 solve 时间。更多计时约定见根 README。

## 练习记录

- [ ] 独立实现并通过所有规模
- [ ] 完成至少一种优化并记录前后耗时
- [ ] 能解释边界处理、同步和访存瓶颈

原题单中的知识点、必做版本与面试追问见 [LeetGPU.md](../../LeetGPU.md)。

## 原题单训练要点

核心知识：

- RoPE 数学原理
- Pairwise Rotation
- Position Index
- 数据布局
- sin / cos 预计算
- 算子融合

面试追问：

1. RoPE 为什么使用旋转？
2. RoPE 如何编码相对位置信息？
3. Interleaved 和 NeoX 风格有什么区别？
4. RoPE 可以与哪些算子融合？
5. Decode 时 RoPE 如何处理位置索引？
