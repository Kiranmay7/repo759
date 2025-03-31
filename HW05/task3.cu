#include <random>
#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include "vscale.cuh"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <number>\n";
        return 1;
    }
    int n = std::atoi(argv[1]);
    float a[n],b[n],*dA,*dB,ms; // Host and device arrays

    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);

    std::random_device rd; // Used to seed the generator
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist1(-10.0, 10.0); // Generate numbers between -10.0 and 10.0
    std::uniform_real_distribution<float> dist2(0.0, 1.0); // Generate numbers between 0.0 and 1.0

    for(int i=0;i<n;++i){
        a[i] = dist1(gen);
        b[i] = dist2(gen);
    }
    /*for(int i=0;i<n;++i){
        std::cout<<"a["<<i<<"] = "<<a[i];
        std::cout<<"b["<<i<<"] = "<<b[i];
    }*/

    cudaMalloc((void**)&dA,sizeof(float)*n);
    cudaMalloc((void**)&dB,sizeof(float)*n);
    cudaMemcpy(dA,a,sizeof(float)*n,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,b,sizeof(float)*n,cudaMemcpyHostToDevice);
    const int threadsperblock = 16;
    const int blockspergrid = (n+threadsperblock-1)/threadsperblock;

    cudaEventRecord(start);
    vscale<<<blockspergrid,threadsperblock>>>(dA,dB,n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(b,dB,sizeof(float)*n,cudaMemcpyDeviceToHost);
    /*for(int i=0;i<n;++i){
        std::cout<<b[i]<<" ";
    }*/
    cudaEventElapsedTime(&ms,start,stop);
    std::cout<<"Total Time: "<<ms<<" ms\n";
    std::cout<<b[0]<<std::endl<<b[n-1]<<std::endl;

    cudaFree(dA);
    cudaFree(dB);
    return 0;
}