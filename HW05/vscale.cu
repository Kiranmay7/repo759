#include <iostream>
#include <cuda.h>
#include <vscale.cuh>

__global__ void vscale(const float *a, float *b, unsigned int n){
    int index = threadIdx.x+blockIdx.x*blockDim.x; 
    b[index] = a[index]*b[index];
}