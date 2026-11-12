#pragma once
#include "solution_versions.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Pointer arguments refer to device memory. See README.md for layouts.
extern "C" void solve(const float* x, const float* W_qkv, const float* cos_sin_cache,
                      const int* positions, float* K_cache, float* V_cache, float* Q_out, int B,
                      int d_model, int H_q, int H_kv, int D, int S_max);
