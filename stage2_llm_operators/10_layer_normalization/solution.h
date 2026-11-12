#pragma once
#include "solution_versions.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Pointer arguments refer to device memory. See README.md for layouts.
extern "C" void solve(const float* input, const float* weight, const float* bias, float* output,
                      int N, int C, float eps);
