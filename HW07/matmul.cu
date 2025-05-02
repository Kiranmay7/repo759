#include "matmul.cuh"
#include <cuda.h>
#include <iostream>
template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, unsigned int n, int BLOCK_SIZE) {
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    T Csub = static_cast<T>(0);
    extern __shared__ char shared_mem[];
    T* shA = reinterpret_cast<T*>(shared_mem);
    T* shB = shA + BLOCK_SIZE * BLOCK_SIZE;

    for (int m = 0; m < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
        int aRow = row;
        int aCol = m * BLOCK_SIZE + tx;
        if (aRow < n && aCol < n)
            shA[ty * BLOCK_SIZE + tx] = A[aRow * n + aCol];
        else
            shA[ty * BLOCK_SIZE + tx] = 0;

        int bRow = m * BLOCK_SIZE + ty;
        int bCol = col;
        if (bRow < n && bCol < n)
            shB[ty * BLOCK_SIZE + tx] = B[bRow * n + bCol];
        else
            shB[ty * BLOCK_SIZE + tx] = 0;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += shA[ty * BLOCK_SIZE + k] * shB[k * BLOCK_SIZE + tx];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = Csub;
}

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim){
    int size = n * n * sizeof(int);
    int *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, size);
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&B_d, size);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&C_d, size);

    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    int sharedMemSize = 2 * block_dim * block_dim * sizeof(int);

    matmul_kernel<<<dimGrid, dimBlock, sharedMemSize>>>(A_d, B_d, C_d, n, block_dim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}


__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim){
    int size = n * n * sizeof(float);
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, size);
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&B_d, size);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&C_d, size);

    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    int sharedMemSize = 2 * block_dim * block_dim * sizeof(float);

    matmul_kernel<<<dimGrid, dimBlock, sharedMemSize>>>(A_d, B_d, C_d, n, block_dim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim){
    int size = n * n * sizeof(double);
    double *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, size);
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&B_d, size);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&C_d, size);

    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    int sharedMemSize = 2 * block_dim * block_dim * sizeof(double);

    matmul_kernel<<<dimGrid, dimBlock, sharedMemSize>>>(A_d, B_d, C_d, n, block_dim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
