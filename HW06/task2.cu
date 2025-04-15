#include <iostream>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>
#include "stencil.cuh"

int main(int argc, char* argv[]) {
    // Parse command line arguments
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " n R threads_per_block" << std::endl;
        return 1;
    }

    unsigned int n = std::atoi(argv[1]);
    unsigned int R = std::atoi(argv[2]);
    unsigned int threads_per_block = std::atoi(argv[3]);

    // Allocate host memory
    float* h_image = new float[n];
    float* h_mask = new float[2 * R + 1];
    float* h_output = new float[n];

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Fill image and mask with random values in [-1, 1]
    for (unsigned int i = 0; i < n; ++i) {
        h_image[i] = dist(gen);
    }
    for (unsigned int i = 0; i < 2 * R + 1; ++i) {
        h_mask[i] = dist(gen);
    }
    /*for(int j=0;j<n;++j){
            std::cout<<h_image[j]<<" ";
        }
    std::cout<<"\n";
    for(int j=0;j<2*R+1;++j){
            std::cout<<h_mask[j]<<" ";
    }
    std::cout<<"\n";*/
    // Allocate device memory
    float *d_image, *d_mask, *d_output;
    cudaMalloc(&d_image, n * sizeof(float));
    cudaMalloc(&d_mask, (2 * R + 1) * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_image, h_image, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, (2 * R + 1) * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Call stencil function
    stencil(d_image, d_mask, d_output, n, R, threads_per_block);

    // Record stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
    /*for(int j=0;j<n;++j){
            std::cout<<h_output[j]<<" ";
        }
    std::cout<<"\n";*/

    std::cout << h_output[n - 1] << std::endl;
    std::cout << ms << " ms"<< std::endl;

    // Cleanup
    delete[] h_image;
    delete[] h_mask;
    delete[] h_output;
    cudaFree(d_image);
    cudaFree(d_mask);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}