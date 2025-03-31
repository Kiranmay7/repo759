#include <random>
#include <iostream>
#include <cuda.h>
#include <stdio.h>

__global__ void saxpy(long int* dA, int a){
    int index = threadIdx.x+blockIdx.x*blockDim.x;
    dA[index] = a*threadIdx.x+blockIdx.x;
}
int main() {
    int numElements = 16;
    long int hA[numElements],*dA; // Host and device arrays
    std::random_device rd; // Used to seed the generator
    std::mt19937 gen(rd());
    int a = gen()%10;
    cudaMalloc((void**)&dA,sizeof(long int)*numElements);
    saxpy<<<2,8>>>(dA,a);
    cudaMemcpy(hA,dA,sizeof(long int)*numElements,cudaMemcpyDeviceToHost);
    for(int i=0;i<numElements;++i){
        std::cout<<hA[i]<<" ";
    }
    std::cout<<std::endl;
    cudaFree(dA);
    return 0;
}