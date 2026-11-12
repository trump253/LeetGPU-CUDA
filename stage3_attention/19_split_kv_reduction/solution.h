#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Pointer arguments refer to device memory. See README.md for layouts.
extern "C" void solve(const float* partial_out, const float* partial_lse, float* output,
                      int num_splits, int num_heads, int head_dim);
