#include <cuda.h>
#include <iostream>
#include "matmul.cuh"
#include <cmath>
__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n){
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if(tidx >= n || tidy >=n) return;
    float Psum = 0.;
    for(int i = 0; i < n; ++i){
        Psum += A[tidy * n + i] * B[i * n + tidx];
    }
    C[tidy * n + tidx] = Psum;
}
void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block){
    float *dA,*dB,*dC;
    cudaMalloc((void**)&dA,sizeof(float)*n*n);
    cudaMalloc((void**)&dB,sizeof(float)*n*n);
    cudaMalloc((void**)&dC,sizeof(float)*n*n);
    cudaMemcpy(dA,A,sizeof(float)*n*n,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,B,sizeof(float)*n*n,cudaMemcpyHostToDevice);
    const int threads = std::sqrt(threads_per_block);
    const int blocks_per_grid = (n+threads-1)/threads;
    //std::cout<<"Threads = "<<threads<<"\n";
    //std::cout<<"Block_per_grid = "<<blocks_per_grid<<"\n";
    dim3 dimGrid(blocks_per_grid,blocks_per_grid,1);
    dim3 dimBlock(threads,threads);

    matmul_kernel<<<dimGrid,dimBlock>>>(dA,dB,dC,n);
    cudaDeviceSynchronize();
    cudaMemcpy(C,dC,sizeof(float)*n*n,cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}