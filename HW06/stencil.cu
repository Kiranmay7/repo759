#include "stencil.cuh"
#include <iostream>

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R) {
    extern __shared__ float shared_mem[];

    // Shared memory allocation
    float* s_mask = &shared_mem[0];                    // For the mask
    int t = (int) R;
    //std::printf("temp R = %d",t);
    float* s_image = &shared_mem[2 * t + 1];           // For the image elements

    const unsigned int tx = threadIdx.x;
    const unsigned int idx = blockIdx.x * blockDim.x + tx;

    // Load mask into shared memory
    // Every thread loads part of the mask
    if (tx < 2 * t + 1) {
        s_mask[tx] = mask[tx];
    }

    // The section of the image to load into shared memory
    const unsigned int block_start = blockIdx.x * blockDim.x;
    const unsigned int block_end = min(block_start + blockDim.x, n);
    const unsigned int shared_mem_size = blockDim.x + 2 * t;

    // Load image data into shared memory (including halo regions)
    // We need to load blockDim.x + 2*R elements total
    for (unsigned int i = tx; i < shared_mem_size; i += blockDim.x) {
        int global_idx = block_start - t + i;

        if (global_idx < 0 || global_idx >= n) {
            // Out of bounds - use boundary value 1.0
            s_image[i] = 1.0f;
        } else {
            // In bounds - use actual image value
            s_image[i] = image[global_idx];
        }
    }

    // Make sure all shared memory loads are done
    __syncthreads();

    // Calculate convolution only for valid thread indices
    if (idx < n) {
        float sum = 0.0f;

        // Apply the stencil operation
        for (int j = -t; j <= t; j++) {
            //std::printf("Inside Stencil compute\n");
            // Position in shared memory: relative to thread's position + offset for halo
            int s_idx = tx + t + j;
            sum += s_image[s_idx] * s_mask[j + t];
        }

        // Write output
        output[idx] = sum;
    }
}

__host__ void stencil(const float* image,
                      const float* mask,
                      float* output,
                      unsigned int n,
                      unsigned int R,
                      unsigned int threads_per_block) {

    unsigned int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    // shared memory size - for mask and image
    size_t shared_mem_size = ((2 * R + 1) + (threads_per_block + 2 * R)) * sizeof(float);
    stencil_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(image, mask, output, n, R);
}