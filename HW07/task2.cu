#include <iostream>
#include <random>
#include <cuda.h>
#include "reduce.cuh"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " N threads_per_block\n";
        return 1;
    }

    unsigned int N = std::atoi(argv[1]);
    unsigned int threads_per_block = std::atoi(argv[2]);

    // Allocate and fill host array
    float *h_input = new float[N];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (unsigned int i = 0; i < N; ++i) {
        h_input[i] = dist(gen);
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    unsigned int blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    cudaMalloc((void **)&d_output, blocks * sizeof(float));

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Call reduce function
    reduce(&d_input, &d_output, N, threads_per_block);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy result back to host
    float result;
    cudaMemcpy(&result, d_input, sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Sum: " << result << std::endl;
    std::cout << "Time: " << ms << " ms" << std::endl;

    // Cleanup
    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}