#include "reduce.cuh"
#include <cuda.h>
#include <iostream>

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Load elements into shared memory
    sdata[tid] = (i < n ? g_idata[i] : 0) + (i + blockDim.x < n ? g_idata[i + blockDim.x] : 0);
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block) {
    unsigned int blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    unsigned int shared_mem_size = threads_per_block * sizeof(float);

    while (N > 1) {
        reduce_kernel<<<blocks, threads_per_block, shared_mem_size>>>(*input, *output, N);
        cudaDeviceSynchronize();

        // Prepare for the next iteration
        N = blocks;
        blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);

        // Swap input and output pointers
        std::swap(*input, *output);
    }
}