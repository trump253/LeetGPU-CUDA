# CUDA 算子手撕面试：LeetGPU 精选题单与 4 周训练计划

> 定位：面向 CUDA 高性能算子 / LLM 推理引擎实习与校招。优先 CUDA C++，Triton 作为补充。
> 
> 使用方式：每道题先不借助 AI 独立作答，再校验正确性、解释优化思路；将题目完成状态、耗时、失败原因记录在此文档。

## 为什么这样安排

LeetGPU 的难度标签不等于面试价值。对已有 Decode Attention 和 W4A16 项目经验的候选人，核心考察更偏向“是否真正能独立写、说清正确性与性能、承担项目相关追问”。本题单按 **基础并行模式 → LLM 常用算子 → 推理与项目深挖** 排序。

优先级：★★★ 必须能独立写；★★ 必须理解并练习；★ 选做。


### 题目链接说明（2026-11-10 核对）

每道题均提供两个入口：**「LeetGPU」** 指向在线练习页面，**「官方题面」** 指向 [AlphaGPU/leetgpu-challenges](https://github.com/AlphaGPU/leetgpu-challenges) 中对应的 `challenge.html`（可查看完整题目、约束及接口，不必先看参考实现）。题目及文件路径已对照官方仓库核对。LeetGPU 网站采用 JavaScript 单页应用，部分深链无法从外部浏览器工具逐个验证；如果某道题的 LeetGPU 深链无法打开，请点击旁边的官方题面，或从 [题库主页](https://leetgpu.com/challenges) 搜索表中英文题名。

## A. CUDA 并行基本功（7 题）

| 完成 | 题目直达链接（LeetGPU / 官方题面） | 优先级 | 必须能讲清的点 | 自设训练目标 |
|---|---|---|---|---|
| [ ] | [Vector Addition](https://leetgpu.com/challenges/vector-addition) · [官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/easy/1_vector_add/challenge.html) | ★★ | Grid/Block/Thread 索引、尾部 mask、合并访存 | 15 分钟正确实现 |
| [ ] | [Matrix Transpose](https://leetgpu.com/challenges/matrix-transpose) · [官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/easy/3_matrix_transpose/challenge.html) | ★★★ | naive 非合并写、shared tile、padding 解决 bank conflict | 30 分钟正确实现 tiled 版本 |
| [ ] | [Reduction](https://leetgpu.com/challenges/reduction) · [官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/4_reduction/challenge.html) | ★★★ | block 树形归约、warp shuffle、多 block 合并 | 30 分钟正确实现 |
| [ ] | [Dot Product](https://leetgpu.com/challenges/dot-product) · [官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/17_dot_product/challenge.html) | ★★ | 归约、分块累加、数值误差 | 25 分钟正确实现 |
| [ ] | [Prefix Sum](https://leetgpu.com/challenges/prefix-sum) · [官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/16_prefix_sum/challenge.html) | ★★ | inclusive/exclusive scan、块内与跨块 | 40 分钟做出基本版本 |
| [ ] | [Histogramming](https://leetgpu.com/challenges/histogramming) · [官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/13_histogramming/challenge.html) | ★ | 原子操作、竞争、block 局部直方图 | 能说明两种实现权衡 |
| [ ] | [General Matrix Multiplication (GEMM)](https://leetgpu.com/challenges/general-matrix-multiplication-gemm) · [官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/22_gemm/challenge.html) | ★★★ | naive、shared-memory tiling、计算/访存比、同步 | 45–60 分钟正确 tiled GEMM |

## B. LLM 常用算子（8 题）

| 完成 | 题目直达链接（LeetGPU / 官方题面） | 优先级 | 必须能讲清的点 | 自设训练目标 |
|---|---|---|---|---|
| [ ] | [Softmax](https://leetgpu.com/challenges/softmax) · [官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/5_softmax/challenge.html) | ★★★ | 减去 max 的数值稳定性、两次归约、行级映射 | 30 分钟独立写稳定版本 |
| [ ] | [RMS Normalization](https://leetgpu.com/challenges/rms-normalization) · [官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/50_rms_normalization/challenge.html) | ★★★ | 平方和归约、rsqrt、eps、融合写回 | 30 分钟正确实现 |
| [ ] | [Layer Normalization](https://leetgpu.com/challenges/layer-normalization) · [官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/113_layer_normalization/challenge.html) | ★★ | mean/variance、归约与 RMSNorm 区别 | 35 分钟正确实现 |
| [ ] | [Rotary Positional Embedding](https://leetgpu.com/challenges/rotary-positional-embedding) · [官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/61_rope_embedding/challenge.html) | ★★★ | 成对维度旋转、位置索引、数据布局 | 30 分钟正确实现 |
| [ ] | [SwiGLU MLP Block](https://leetgpu.com/challenges/swiglu-mlp-block) · [官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/84_swiglu_mlp_block/challenge.html) | ★★ | SiLU、逐元素 gating、多 kernel/融合选择 | 先正确后说明融合边界 |
| [ ] | [Fused Residual Add and RMS Norm](https://leetgpu.com/challenges/fused-residual-add-and-rms-norm) · [官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/83_fused_residual_add_rms_norm/challenge.html) | ★★★ | 读写次数、residual + norm 融合、临时张量消除 | 40 分钟完成正确融合 |
| [ ] | [Weight Dequantization](https://leetgpu.com/challenges/weight-dequantization) · [官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/64_weight_dequantization/challenge.html) | ★★ | scale/zero-point、量化格式、pack/unpack | 30 分钟实现基本反量化 |
| [ ] | [INT4 Weight-Only Quantized MatMul](https://leetgpu.com/challenges/int4-weight-only-quantized-matmul) · [官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/81_int4_matmul/challenge.html) | ★★★ | W4A16、bit unpack、反量化与计算融合 | 解释并能手写简化版，不要求限时工业优化 |

## C. LLM 推理引擎 / 项目深挖（6 题）

| 完成 | 题目直达链接（LeetGPU / 官方题面） | 优先级 | 必须能讲清的点 | 自设训练目标 |
|---|---|---|---|---|
| [ ] | [Softmax Attention](https://leetgpu.com/challenges/softmax-attention) · [官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/6_softmax_attention/challenge.html) | ★★★ | QK^T、mask、softmax、乘 V、数值稳定 | 能写功能正确的简化版本 |
| [ ] | [Grouped Query Attention](https://leetgpu.com/challenges/grouped-query-attention) · [官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/80_grouped_query_attention/challenge.html) | ★★★ | Q head 到 KV head 映射、带宽、GQA vs MHA/MQA | 40 分钟做简化 decode 版本 |
| [ ] | [Paged KV-Cache Attention](https://leetgpu.com/challenges/paged-kv-cache-attention) · [官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/86_paged_attention/challenge.html) | ★★★ | block table、物理页映射、非连续访存、边界 | 基于项目原理从空白复写核心索引 |
| [ ] | [Split-KV Attention Reduction](https://leetgpu.com/challenges/split-kv-attention-reduction) · [官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/128_split_kv_attention_reduction/challenge.html) | ★★ | 分段 online-softmax 的 (m,l,o) 合并 | 手写分段结果合并逻辑 |
| [ ] | [Flash Decoding Attention](https://leetgpu.com/challenges/flash-decoding-attention) · [官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/hard/141_flash_decoding_attention/challenge.html) | ★★ | decode 长上下文、split-KV、parallelism | 画出数据流并写核心计算框架 |
| [ ] | [MoE Top-K Gating](https://leetgpu.com/challenges/moe-top-k-gating) · [官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/67_moe_topk_gating/challenge.html) | ★ | Top-K 选择、routing、访存与分支 | 写正确基础版、讲优化方案 |

**选做扩展**：[2D Convolution](https://leetgpu.com/challenges/2d-convolution)（[官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/10_2d_convolution/challenge.html)）、[Top K Selection](https://leetgpu.com/challenges/top-k-selection)（[官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/29_top_k_selection/challenge.html)）、[Grouped GEMM](https://leetgpu.com/challenges/grouped-gemm)（[官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/99_grouped_gemm/challenge.html)）、[Fused QKV Projection with RoPE and KV Cache Update](https://leetgpu.com/challenges/fused-qkv-projection-with-rope-and-kv-cache-update)（[官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/115_fused_qkv_rope_kv_cache_update/challenge.html)）、[Causal Self-Attention](https://leetgpu.com/challenges/causal-self-attention)（[官方题面](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/hard/53_casual_attention/challenge.html)）。除非目标 JD 明确强调这些方向，不必在第一轮全部刷完。

## 最优先 12 道（面试时间有限时）

1. [Matrix Transpose](https://leetgpu.com/challenges/matrix-transpose)
2. [Reduction](https://leetgpu.com/challenges/reduction)
3. [General Matrix Multiplication (GEMM)](https://leetgpu.com/challenges/general-matrix-multiplication-gemm)
4. [Softmax](https://leetgpu.com/challenges/softmax)
5. [RMS Normalization](https://leetgpu.com/challenges/rms-normalization)
6. [Layer Normalization](https://leetgpu.com/challenges/layer-normalization)
7. [Rotary Positional Embedding](https://leetgpu.com/challenges/rotary-positional-embedding)
8. [Fused Residual Add and RMS Norm](https://leetgpu.com/challenges/fused-residual-add-and-rms-norm)
9. [Softmax Attention](https://leetgpu.com/challenges/softmax-attention)
10. [Grouped Query Attention](https://leetgpu.com/challenges/grouped-query-attention)
11. [Paged KV-Cache Attention](https://leetgpu.com/challenges/paged-kv-cache-attention)
12. [INT4 Weight-Only Quantized MatMul](https://leetgpu.com/challenges/int4-weight-only-quantized-matmul)

## 4 周节奏（每天约 60–90 分钟，可适当延长难题）

| 周 | 任务 | 检查点 |
|---|---|---|
| 第 1 周 | A 组基础题；Reduction/Transpose/GEMM 必须复写 | 能手画线程映射和访存图 |
| 第 2 周 | B 组 LLM 基础算子；优先 Softmax、RMSNorm、RoPE | 能解释数值稳定、融合收益和读写次数 |
| 第 3 周 | C 组推理引擎题；关联自己 Decode Attention / W4A16 代码 | 能在白板上复述数据流并处理 shape/stride |
| 第 4 周 | 每隔 1–2 天随机抽题 45–60 分钟限时实战；剩余时间复盘薄弱项 | 盲写成功率与讲解质量优于刷题数 |

## 单题作答流程（模拟面试而不是复制提交）

- **0–5 分钟：** 确认输入 shape、dtype、stride、layout、输出语义和精度要求；画线程映射。
- **5–35 分钟：** 无 AI、无旧代码，独立写 CUDA C++ 正确版本；先正确再优化。
- **35–45 分钟：** 跑正确性：奇数长度、小/大 shape、非整倍数、极值、数值误差、是否越界。
- **45–60 分钟：** 口述优化：global memory 读写量、coalescing、shared memory、warp-level primitives、occupancy、launch overhead、profile 方法。

复杂题（GEMM、Paged Attention、Flash Decoding）可以单独用 60–120 分钟，不能把自设训练目标理解为实际面试统一时限。

## 每题复盘卡片（复制使用）

```md
### 题目：
- 日期：
- 首次独立完成：是 / 否
- 首次正确实现用时： 分钟
- CUDA C++：是否完成；Triton：是否完成
- 核心公式与线程映射：
- 易错点（索引、同步、精度、边界）：
- 基线：shape / dtype / GPU / ms 或 us
- 优化版：shape / dtype / GPU / ms 或 us
- 实测加速比（同设备、同输入、同计时方法）：
- Nsight 发现的瓶颈：
- 三个面试追问及自己的回答：
- 复刷日期（建议 D+1 / D+3 / D+7）：
```

> 重要：性能测试必须固定 GPU、shape、dtype、warmup 和计时口径；不可直接把 RTX 3090 与 RTX 5060 Ti 的绝对延迟作为算法优劣依据。

## 本地 Agent 面试官 Prompt（可直接复制）

```text
你是国内大厂 CUDA 高性能算子 / LLM 推理开发岗位的技术面试官。请围绕 CUDA C++ 算子手撕面试训练我。每轮从本题单随机选择一道题（优先 ★★★），先只给题目、函数接口、输入输出 shape/dtype/stride、样例、约束与正确性标准，不给任何实现、伪代码或提示。待我提交自己的实现后，再分阶段检查：1）是否编译、是否正确（边界/同步/精度）；2）线程映射与访存；3）可落地的两步性能优化；4）结合我的 Decode Attention、Paged KV Cache、W4A16 项目进行深入追问。若我要求提示，按从弱到强三个等级给出。最后按正确性 40%、CUDA 原理 25%、性能分析 20%、表达与独立性 15% 打分，输出补强建议。不要代写第一版代码。
```

## 资源与选择

| 资源 | 链接 | 适用场景 |
|---|---|---|
| LeetGPU | https://leetgpu.com/challenges | 在线写 CUDA/Triton、过正确性、看平台成绩；本训练计划主站 |
| GPU MODE / Reference Kernels | https://github.com/gpu-mode/reference-kernels | 有固定测试和优化约束的深度性能挑战；适合面试进阶/性能工程 |
| GPU Puzzles | https://github.com/srush/GPU-Puzzles | 基础并行思维教学，基于 Numba；查漏补缺而非主要刷题 |
| Triton Puzzles | https://github.com/gpu-mode/Triton-Puzzles | 系统练 Triton 的 load/store、mask、block 编程 |
| LeetsCUDA | https://www.leetscuda.com/problems | 包含 CUDA 和 AI Infra 题型；截至 2026-11 检索时，其 Matrix Transpose 页面提示 judge 维护中 |

## 通过标准（自设）

- **独立写：** Reduction、Transpose、Softmax、RMSNorm、RoPE 不借助答案能盲写。
- **能定位：** 对错误能定位 race、越界、数值精度、warp/block 同步问题。
- **能解释：** 能说出优化前后全局内存访问和 GPU 资源占用为何变化。
- **能迁移：** 将 Softmax/Reduction 的知识迁移到 Decode Attention；将 tile/量化知识迁移到 W4A16。
- **能诚实表达贡献：** 分清亲自实现与 Agent 辅助完成的代码，对个人负责的部分能够独立复现和解释。
- **参考答案：** [答案](https://github.com/LLAA178/LeetGPU-Guidebook#vector-addition)
