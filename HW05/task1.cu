#include <stdio.h>
#include <cuda.h>
#include <iostream>
__global__ void factorial(){
    int fact = 1;
    for(int i = 1; i<=threadIdx.x+1; i++){
        fact = fact*i;
    }
    std::printf("%d!=%d\n", threadIdx.x+1, fact);
}

int main(){
    factorial<<<1,8>>>();
    cudaDeviceSynchronize();
    return 0;
}