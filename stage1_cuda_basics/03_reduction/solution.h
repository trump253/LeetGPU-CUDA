#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Pointer arguments refer to device memory. See README.md for layouts.
extern "C" void solve(const float* input, float* output, int N);
