#include "solution.h"

__global__ void histogramming_kernel(const int* input, int* histogram, int N, int num_bins) {
    // TODO: implement your GPU kernel here.
}

extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    // TODO: choose a grid/block mapping for this operator.
    histogramming_kernel<<<1, 256>>>(input, histogram, N, num_bins);
    // The harness checks CUDA errors and synchronizes; no sync is needed here.
}
