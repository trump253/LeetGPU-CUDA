# CUDA 算子手撕面试题单

> 适用方向：CUDA 算子开发 / LLM 推理引擎 / AI Infra
>
> 编程语言：CUDA C++ 为主，Triton 为辅
>
> 训练平台：https://leetgpu.com/challenges
>
> 官方题库：https://github.com/AlphaGPU/leetgpu-challenges
>
> 训练周期：4 周
>
> 题目规模：21 道核心题 + 5 道进阶题

---

## 一、训练目标

本题单不是为了刷 LeetGPU 排行榜，而是为了应对国内大厂 CUDA 算子开发和 LLM 推理岗位的技术面试。

最终目标：

1. 能闭卷手写常见 CUDA Kernel。
2. 能解释 Thread / Warp / Block 的任务划分。
3. 熟练掌握 Global Memory、Shared Memory、Register 的使用。
4. 掌握 Reduction、Tiling、Coalescing、Warp Shuffle。
5. 能手写 Softmax、RMSNorm、RoPE、GEMM。
6. 能理解并实现简化版 Attention。
7. 能解释 Kernel Fusion、算术强度、访存优化。
8. 能将算子知识与 CUDALM-Core、NibbleGEMM 项目关联。

### 优先级说明

- P0：必须闭卷独立写出来。
- P1：需要会写基础版本，并能解释优化思路。
- P2：掌握原理，时间充裕时完成。

### 完成标准

每道题分为三个级别：

- Level 1：正确实现，能够通过测试。
- Level 2：至少完成一项有意义的 CUDA 优化。
- Level 3：能够解释性能瓶颈，并接受面试官追问。

建议基础题控制在 20～40 分钟。

GEMM、Attention 等复杂题可延长至 60～90 分钟。

---

# 二、第一阶段：CUDA 基本功

目标：线程映射、访存、归约、共享内存。

## 01. Vector Addition

优先级：P1
难度：Easy

LeetGPU：
https://leetgpu.com/challenges/vector-addition

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/easy/1_vector_add/challenge.html

核心知识：

- threadIdx / blockIdx / blockDim
- Grid 与 Block 配置
- Global Memory 访问
- 边界检查
- Memory Coalescing

面试追问：

1. 为什么通常选择 256 threads/block？
2. 什么时候会出现非合并访存？
3. 为什么 Vector Add 通常受内存带宽限制？
4. 如何估算理论最低执行时间？

完成状态：- [ ]

## 02. Matrix Transpose

优先级：P0
难度：Easy

LeetGPU：
https://leetgpu.com/challenges/matrix-transpose

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/easy/3_matrix_transpose/challenge.html

核心知识：

- Row-major 地址映射
- Global Memory Coalescing
- Shared Memory Tiling
- Bank Conflict
- Shared Memory Padding

必须实现：

1. Naive Transpose
2. Shared Memory Tiled Transpose
3. Padding 优化版本

面试追问：

1. 为什么直接转置可能造成非合并访存？
2. 为什么 Shared Memory 可以提高性能？
3. 为什么 tile[32][33] 常被使用？
4. Bank Conflict 是什么？

完成状态：- [ ]

## 03. Reduction

优先级：P0
难度：Medium

LeetGPU：
https://leetgpu.com/challenges/reduction

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/4_reduction/challenge.html

核心知识：

- Parallel Reduction
- Shared Memory
- __syncthreads()
- Warp Shuffle
- 多 Block 归约
- AtomicAdd

必须实现：

1. Shared Memory Tree Reduction
2. Warp Shuffle Reduction
3. 多 Block Reduction

面试追问：

1. Reduction 为什么需要同步？
2. Warp Shuffle 为什么不需要 Shared Memory？
3. 多 Block 如何得到最终结果？
4. AtomicAdd 有哪些性能问题？
5. 如何处理非 2 的幂长度？

完成状态：- [ ]

## 04. Dot Product

优先级：P1
难度：Medium

LeetGPU：
https://leetgpu.com/challenges/dot-product

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/17_dot_product/challenge.html

核心知识：

- Element-wise Multiplication
- Reduction
- Warp-level Reduction
- 数据并行
- 浮点累加误差

面试追问：

1. Dot Product 和 Reduction 有什么不同？
2. 如何减少 Global Memory 读写？
3. float32 累加有什么精度问题？
4. 如何使用 Warp Shuffle 实现？

完成状态：- [ ]

## 05. Prefix Sum

优先级：P1
难度：Medium

LeetGPU：
https://leetgpu.com/challenges/prefix-sum

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/16_prefix_sum/challenge.html

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

完成状态：- [ ]

## 06. Histogramming

优先级：P2
难度：Medium

LeetGPU：
https://leetgpu.com/challenges/histogramming

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/13_histogramming/challenge.html

核心知识：

- AtomicAdd
- 数据竞争
- Shared Memory Histogram
- 局部直方图合并
- Atomic Contention

面试追问：

1. 为什么多个线程更新同一个 Bin 会产生竞争？
2. 如何减少 Atomic 操作？
3. Shared Memory Histogram 为什么可能更快？

完成状态：- [ ]

## 07. General Matrix Multiplication (GEMM)

优先级：P0
难度：Medium

LeetGPU：
https://leetgpu.com/challenges/general-matrix-multiplication-gemm

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/22_gemm/challenge.html

核心知识：

- Matrix Multiplication
- Shared Memory Tiling
- Data Reuse
- Arithmetic Intensity
- Register Blocking
- Thread Coarsening

必须实现：

1. Naive GEMM
2. Shared Memory Tiled GEMM
3. Register Blocking 优化版本（进阶）

面试追问：

1. 为什么 Tiling 可以提高性能？
2. 如何计算 GEMM 的 FLOPs？
3. GEMM 为什么适合 GPU？
4. 如何估算数据搬运量？
5. 什么是 Arithmetic Intensity？
6. Tensor Core 与 CUDA Core 的区别是什么？
7. 为什么工业级 GEMM 通常需要更复杂的优化？

完成状态：- [ ]

---

# 三、第二阶段：LLM 常用算子

目标：掌握推理引擎中的基础算子和 Kernel Fusion。

## 08. Softmax

优先级：P0
难度：Medium

LeetGPU：
https://leetgpu.com/challenges/softmax

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/5_softmax/challenge.html

核心知识：

- Max Reduction
- Sum Reduction
- Numerical Stability
- Warp Shuffle
- Online Softmax

必须实现：

1. 基础 Softmax
2. 数值稳定版本
3. Warp / Block 并行优化

面试追问：

1. 为什么 Softmax 要减去最大值？
2. Softmax 为什么需要两次 Reduction？
3. 如何减少 Global Memory 访问？
4. Online Softmax 是什么？
5. FlashAttention 如何利用 Online Softmax？

完成状态：- [ ]

## 09. RMS Normalization

优先级：P0
难度：Medium

LeetGPU：
https://leetgpu.com/challenges/rms-normalization

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/50_rms_normalization/challenge.html

核心知识：

- Reduction
- rsqrt
- Warp Shuffle
- Kernel Fusion
- Hidden Dimension 并行

必须实现：

1. 基础 RMSNorm
2. 单 Kernel 融合版本
3. Warp Reduction 优化版本

面试追问：

1. RMSNorm 和 LayerNorm 的区别？
2. 为什么 RMSNorm 不需要计算均值？
3. eps 的作用是什么？
4. 如何融合 Reduction 和 Scale？
5. RMSNorm 为什么经常是 Memory-bound？

完成状态：- [ ]

## 10. Layer Normalization

优先级：P1
难度：Medium

LeetGPU：
https://leetgpu.com/challenges/layer-normalization

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/113_layer_normalization/challenge.html

核心知识：

- Mean Reduction
- Variance Reduction
- Numerical Stability
- Welford Algorithm
- Kernel Fusion

面试追问：

1. LayerNorm 与 BatchNorm 有什么区别？
2. 如何计算均值与方差？
3. Welford Algorithm 的优势是什么？
4. 为什么 RMSNorm 更适合某些 LLM 架构？

完成状态：- [ ]

## 11. Rotary Positional Embedding (RoPE)

优先级：P0
难度：Medium

LeetGPU：
https://leetgpu.com/challenges/rotary-positional-embedding

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/61_rope_embedding/challenge.html

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

完成状态：- [ ]

## 12. SwiGLU MLP Block

优先级：P1
难度：Medium

LeetGPU：
https://leetgpu.com/challenges/swiglu-mlp-block

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/84_swiglu_mlp_block/challenge.html

核心知识：

- SiLU
- Gating
- GEMM
- Element-wise Fusion
- 中间张量访存

面试追问：

1. SwiGLU 和普通 ReLU MLP 有什么区别？
2. 哪些操作可以融合？
3. 为什么不一定要把整个 MLP 融合成单 Kernel？
4. 如何判断融合是否划算？

完成状态：- [ ]

## 13. Fused Residual Add and RMS Norm

优先级：P0
难度：Medium

LeetGPU：
https://leetgpu.com/challenges/fused-residual-add-and-rms-norm

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/83_fused_residual_add_rms_norm/challenge.html

核心知识：

- Kernel Fusion
- Residual Connection
- RMSNorm
- Global Memory Traffic
- 寄存器中的中间结果

面试追问：

1. 为什么 Residual Add 和 RMSNorm 适合融合？
2. 融合前后显存访问量发生了什么变化？
3. Fusion 是否一定提升性能？
4. Register Pressure 可能带来什么问题？

完成状态：- [ ]

## 14. Weight Dequantization

优先级：P1
难度：Medium

LeetGPU：
https://leetgpu.com/challenges/weight-dequantization

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/64_weight_dequantization/challenge.html

核心知识：

- Quantization
- Dequantization
- Scale / Zero Point
- Bit Packing
- Memory Bandwidth

面试追问：

1. 为什么要进行权重量化？
2. Per-channel 与 Per-group 量化的区别？
3. Dequantization 为什么可能成为瓶颈？
4. 如何减少反量化中间张量？

完成状态：- [ ]

## 15. INT4 Weight-Only Quantized MatMul

优先级：P0
难度：Medium

LeetGPU：
https://leetgpu.com/challenges/int4-weight-only-quantized-matmul

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/81_int4_matmul/challenge.html

核心知识：

- W4A16
- Bit Unpacking
- Weight-only Quantization
- Dequantization Fusion
- GEMM
- Quantization Group

面试追问：

1. W4A16 是什么意思？
2. INT4 权重如何存储？
3. 如何从 uint32 中提取 INT4？
4. 为什么不先反量化完整权重矩阵？
5. 为什么小 Batch 可能更适合量化 GEMV？
6. 量化计算什么时候会从 Memory-bound 变成 Compute-bound？

完成状态：- [ ]

---

# 四、第三阶段：LLM 推理与 Attention

目标：能够解释并实现简化版推理算子。

## 16. Softmax Attention

优先级：P0
难度：Medium

LeetGPU：
https://leetgpu.com/challenges/softmax-attention

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/6_softmax_attention/challenge.html

核心知识：

- QK^T
- Softmax
- Attention Score
- Scaling
- Attention Fusion

面试追问：

1. Attention 的完整计算流程？
2. Prefill 和 Decode 的区别？
3. 为什么 Attention 可能产生较大的中间矩阵？
4. FlashAttention 解决了什么问题？
5. Online Softmax 如何避免存储完整 Attention Matrix？

完成状态：- [ ]

## 17. Grouped Query Attention (GQA)

优先级：P0
难度：Medium

LeetGPU：
https://leetgpu.com/challenges/grouped-query-attention

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/80_grouped_query_attention/challenge.html

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

完成状态：- [ ]

## 18. Paged KV-Cache Attention

优先级：P0
难度：Medium

LeetGPU：
https://leetgpu.com/challenges/paged-kv-cache-attention

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/86_paged_attention/challenge.html

核心知识：

- Paged KV Cache
- Block Table
- Physical Block
- Logical Block
- Decode Attention
- 非连续访存

面试追问：

1. PagedAttention 为什么需要 Block Table？
2. 逻辑 Token 如何映射到物理地址？
3. Paged KV Cache 解决了什么问题？
4. 为什么 Decode Attention 经常 Memory-bound？
5. 如何处理不同长度的 Sequence？
6. 为什么 Paged KV Cache 可能带来非连续访存？

完成状态：- [ ]

## 19. Split-KV Attention Reduction

优先级：P1
难度：Medium

LeetGPU：
https://leetgpu.com/challenges/split-kv-attention-reduction

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/128_split_kv_attention_reduction/challenge.html

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

完成状态：- [ ]

## 20. Flash Decoding Attention

优先级：P1
难度：Hard

LeetGPU：
https://leetgpu.com/challenges/flash-decoding-attention

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/hard/141_flash_decoding_attention/challenge.html

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

完成状态：- [ ]

## 21. MoE Top-K Gating

优先级：P2
难度：Medium

LeetGPU：
https://leetgpu.com/challenges/moe-top-k-gating

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/67_moe_topk_gating/challenge.html

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

完成状态：- [ ]

---

# 五、进阶题（选做）

完成核心题后，根据岗位方向进行专项训练。

## 22. Fused QKV Projection with RoPE and KV Cache Update

方向：LLM 推理引擎

LeetGPU：
https://leetgpu.com/challenges/fused-qkv-projection-with-rope-and-kv-cache-update

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/115_fused_qkv_rope_kv_cache_update/challenge.html

核心考点：
QKV Projection、RoPE、KV Cache 写入、Kernel Fusion。

## 23. INT8 KV-Cache Attention

方向：LLM 推理 / 量化

LeetGPU：
https://leetgpu.com/challenges/int8-kv-cache-attention

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/96_int8_kv_cache_attention/challenge.html

核心考点：
INT8 KV Cache、反量化、带宽优化、量化误差。

## 24. Grouped GEMM

方向：高性能算子 / MoE

LeetGPU：
https://leetgpu.com/challenges/grouped-gemm

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/99_grouped_gemm/challenge.html

核心考点：
Grouped MatMul、异构矩阵尺寸、任务调度、负载均衡。

## 25. Top K Selection

方向：GPU 并行算法

LeetGPU：
https://leetgpu.com/challenges/top-k-selection

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/29_top_k_selection/challenge.html

核心考点：
Parallel Selection、Warp 协作、Sorting、Top-K。

## 26. Softmax Attention Backward

方向：训练算子 / FlashAttention

LeetGPU：
https://leetgpu.com/challenges/softmax-attention-backward

官方题面：
https://github.com/AlphaGPU/leetgpu-challenges/blob/main/challenges/medium/111_softmax_attention_backward/challenge.html

核心考点：
反向传播、Attention Gradient、Reduction、数值稳定性。

---

# 六、优先掌握的 12 道面试题

如果面试时间紧张，优先刷以下 12 道。

| 顺序 | 题目                     | 训练重点             |
| ---- | ------------------------ | -------------------- |
| 1    | Reduction                | Warp Shuffle         |
| 2    | Matrix Transpose         | 合并访存             |
| 3    | Softmax                  | 数值稳定             |
| 4    | RMSNorm                  | Reduction + Fusion   |
| 5    | GEMM                     | Shared Memory Tiling |
| 6    | LayerNorm                | 均值与方差           |
| 7    | RoPE                     | 索引与数据布局       |
| 8    | Fused Residual RMSNorm   | Kernel Fusion        |
| 9    | INT4 MatMul              | 量化                 |
| 10   | Softmax Attention        | Attention 计算       |
| 11   | GQA                      | Head 映射            |
| 12   | Paged KV-Cache Attention | Cache 地址映射       |

掌握标准：

- 前 8 道：能够闭卷独立实现。
- 后 4 道：能够实现正确的简化版本，并解释关键优化。

---

# 七、四周刷题计划

## Week 1：CUDA 基本功

- Day 1：Vector Addition + Reduction
- Day 2：Matrix Transpose
- Day 3：Dot Product + Histogram
- Day 4：Prefix Sum
- Day 5：GEMM Naive
- Day 6：GEMM Tiling
- Day 7：复盘 + 随机抽题

本周目标：

- 理解 CUDA 线程模型。
- 掌握 Shared Memory。
- 掌握 Warp Shuffle。
- 能够解释 Coalescing、Bank Conflict 和 Tiling。

## Week 2：LLM 基础算子

- Day 8：Softmax
- Day 9：RMSNorm
- Day 10：LayerNorm
- Day 11：RoPE
- Day 12：SwiGLU + Fused Residual RMSNorm
- Day 13：Weight Dequantization + INT4 MatMul
- Day 14：复盘 + 模拟面试

本周目标：

- 能独立实现常见 LLM 算子。
- 理解数值稳定性。
- 掌握基本 Kernel Fusion。
- 理解 Weight-only Quantization。

## Week 3：Attention 专项

- Day 15：Softmax Attention
- Day 16：GQA
- Day 17：Paged KV-Cache Attention
- Day 18：Split-KV Attention Reduction
- Day 19：Flash Decoding
- Day 20：MoE Top-K Gating
- Day 21：Attention 专项模拟面试

本周目标：

- 理解 Prefill / Decode。
- 掌握 GQA Head 映射。
- 掌握 Paged KV Cache。
- 理解 Split-KV 与 Online Softmax。
- 能解释项目中的 Attention 优化设计。

## Week 4：面试冲刺

- Day 22：随机抽取 2 道基础题
- Day 23：随机抽取 2 道 LLM 算子题
- Day 24：随机抽取 1 道 Attention 题
- Day 25：复盘 GEMM / INT4
- Day 26：复盘 Softmax / RMSNorm / RoPE
- Day 27：完整模拟面试
- Day 28：整理错题、薄弱点与项目追问

进阶题作为选做内容，不要求全部完成。

本周目标：

- 基础题能够在限定时间内独立完成。
- 复杂题能够清楚解释设计与优化。
- 面对性能追问能够给出合理分析。
- 避免只会背诵现成 Kernel。

---

# 八、每道题的三轮训练法

## Round 1：闭卷实现

要求：

- 不看答案。
- 不让 Agent 写代码。
- 不参考过去的项目实现。
- 允许查看 CUDA API 文档。
- 优先保证正确性。

完成后记录：

- 实际耗时。
- 是否一次通过。
- 编译错误。
- 正确性错误。
- 哪些地方依赖提示。

## Round 2：性能优化

在正确版本基础上考虑：

1. 是否存在非合并访存？
2. 是否需要 Shared Memory？
3. 是否存在重复 Global Memory 读取？
4. 是否可以使用 Warp Shuffle？
5. 是否存在不必要的 Kernel Launch？
6. 是否可以 Fusion？
7. 是否存在较大的中间张量？
8. 是否可能存在 Register Pressure？
9. 当前算子主要受计算还是访存限制？

要求至少选择一项进行优化，并使用测试数据验证。

## Round 3：模拟面试

由本地 Agent 担任面试官。

面试内容包括：

- 原理解释。
- Kernel 实现。
- 边界条件。
- 同步问题。
- 访存分析。
- 性能优化。
- 项目关联。

重点不是背诵答案，而是能够解释自己的设计。

---

# 九、刷题记录模板

## 题目名称

日期：

题目链接：

优先级：

### 1. 正确性

- [ ] 独立完成
- [ ] 编译通过
- [ ] 正确性测试通过
- [ ] 覆盖边界条件

### 2. 性能

- Baseline 耗时：
- 优化版本耗时：
- Speedup：
- 测试 GPU：
- 数据规模：
- 数据类型：

注意：只比较相同设备、相同输入规模、相同测试方法下的性能。

### 3. CUDA 知识点

涉及的技术：

- Thread Mapping：
- Memory Coalescing：
- Shared Memory：
- Warp Shuffle：
- Synchronization：
- Kernel Fusion：

### 4. 面试追问

Q1：

A1：

Q2：

A2：

Q3：

A3：

### 5. 错误和复盘

遇到的问题：

原因分析：

解决方法：

需要复习的知识点：

### 6. 完成评级

- [ ] Level 1：正确
- [ ] Level 2：完成优化
- [ ] Level 3：能够应对面试追问

下次复习日期：

---

# 十、本地 Agent 模拟面试 Prompt

你现在是我的 CUDA 算子开发岗位模拟面试官。

我的目标岗位：

- 国内大厂 CUDA 算子开发
- LLM 推理引擎开发
- AI Infra

我的背景：

- 已开发 CUDALM-Core 推理引擎。
- 学习过 CUDA Decode Attention。
- 研究过 W4A16 量化 GEMM。
- 现在希望训练闭卷 CUDA Kernel 手撕能力。

请按照以下规则进行面试。

## 规则 1：随机出题

从当前 Markdown 题单中选择一道题。

默认优先 P0 和 P1。

不要提前透露实现方法。

## 规则 2：提供面试题目

提供：

1. 算子功能描述。
2. 输入输出接口。
3. 数据类型。
4. Tensor Shape。
5. 正确性要求。
6. 关键边界条件。
7. 时间限制。

优先提供 CUDA C++ 接口。

## 规则 3：不要替我实现

我编写代码期间：

- 不要直接给出完整实现。
- 不要自动替我修改源码。
- 不要未经要求生成最终答案。
- 可以根据我的提问给予逐步提示。

## 规则 4：代码 Review

完成后进行代码 Review，检查：

1. 正确性。
2. 越界访问。
3. Race Condition。
4. __syncthreads() 使用。
5. Global Memory 访问。
6. Shared Memory 使用。
7. Warp 协作。
8. 数值稳定性。
9. 性能瓶颈。

不要运行大规模耗时测试，除非我明确要求。

## 规则 5：模拟面试追问

针对我写出的代码，提出 3～5 个技术追问。

例如：

- 为什么这么划分线程？
- 有没有更好的访存方式？
- 如何优化 Reduction？
- 如何降低显存读写？
- 如何提高 Occupancy？
- 是否存在 Bank Conflict？
- 这个 Kernel 为什么可能是 Memory-bound？

根据我的回答继续追问。

## 规则 6：打分

面试结束后，按照以下标准打分：

- 正确性：30 分
- CUDA 基础：20 分
- 性能分析：20 分
- 代码质量：15 分
- 技术表达：15 分

总分：100 分。

给出：

1. 面试评分。
2. 做得好的地方。
3. 暴露的知识盲区。
4. 需要复习的 CUDA 知识。
5. 推荐的下一道题。

## 规则 7：训练原则

目标是提高我的独立实现能力，而不是帮我完成题目。

只在我明确要求时给出参考答案。

请从一道 P0 题目开始进行模拟面试。

---

# 十一、推荐练习平台

## LeetGPU

https://leetgpu.com/challenges

主力刷题平台。

特点：

- CUDA C++ 在线编程
- Triton 等多种 GPU 编程语言
- 正确性验证
- 性能测试

## GPU MODE Reference Kernels

https://github.com/gpu-mode/reference-kernels

用于进一步学习性能优化。

## Triton Puzzles

https://github.com/gpu-mode/Triton-Puzzles

用于训练 Triton 编程。

建议先完成 CUDA C++ 主线，再补充 Triton。

## GPU Puzzles

https://github.com/srush/GPU-Puzzles

适合重新巩固 GPU 线程模型和基础并行编程。

---

# 十二、最终验收标准

完成训练后，应该能够：

1. 20～30 分钟独立实现基本 Reduction。
2. 独立完成 Shared Memory Tiled Transpose。
3. 独立实现数值稳定的 Softmax。
4. 独立实现 RMSNorm。
5. 手写 Naive GEMM 和 Tiled GEMM。
6. 解释 RoPE 的数学与数据布局。
7. 写出基础 Attention 计算流程。
8. 解释 GQA 的 KV Head 映射。
9. 解释 Paged KV Cache 地址转换。
10. 解释 INT4 权重解包和反量化融合。
11. 分析常见 Kernel 的性能瓶颈。
12. 将以上知识与实际推理引擎项目关联。

最重要的验收方式：

随机抽取一道 P0 题，在没有 Agent 代写、没有现成答案的情况下，完成正确实现并解释优化思路。

不要只追求刷题数量。优先确保每道 P0 题都真正掌握。