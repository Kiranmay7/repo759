#include <random>
#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include "matmul.cuh"

void print(float *A,int n){
    for(int i=0;i<n;++i)
    {   for(int j=0;j<n;++j){
            std::cout<<A[i*n+j]<<" ";
        }
        std::cout<<"\n";
    }
}

void mmul_ref(const float* A, const float* B, float* C, const int n){
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
            C[i*n+j] = 0;
            for(int k=0;k<n;++k){
                C[i*n+j]+=A[i*n+k]*B[k*n+j];
            }
        }
    }
}

void MatrixMaxDifference(const float* A, const float* B,const int n)
{
    float result = 0.;
    for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
        result = std::max( result, std::abs( A[i*n+j] - B[i*n+j] ) );
    std::cout << "Discrepancy between two methods : " << result << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <number>\n";
        return 1;
    }
    int n = std::atoi(argv[1]);
    int threads_per_block = std::atoi(argv[2]);
    float A[n*n],B[n*n],C[n*n],Cref[n*n],ms; // Host and device arrays
    //std::cout<<"main() for "<<n<<"\n";
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);

    std::random_device rd; // Used to seed the generator
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist1(-1.0, 1.0); // Generate numbers between -1.0 and 1.0

    for(int i = 0; i<n;++i)
    {   for(int j=0;j<n;++j){
            A[i*n+j] = dist1(gen);
            B[i*n+j] = dist1(gen);
        }
    }
    //std::cout<<"A: \n";
    //print(A,n);
    //std::cout<<"B: \n";
    //print(B,n);
    cudaEventRecord(start);
    matmul(A,B,C,n,threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //std::cout<<"C: \n";
    //print(C,n);
    mmul_ref(A,B,Cref,n);
    //std::cout<<"Cref :\n";
    //print(Cref,n);
    MatrixMaxDifference(C,Cref,n);
    cudaEventElapsedTime(&ms,start,stop);
    std::cout<<"Total Time: "<<ms<<" ms\n";
    std::cout<<C[n*n-1]<<std::endl;

    return 0;
}