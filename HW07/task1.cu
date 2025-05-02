#include <random>
#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include "matmul.cuh"
#include <cmath>

template <typename T>
void fillmatrix(T* A, T* B, int n) {
    std::random_device rd; // Used to seed the generator
    std::mt19937 gen(rd());
    
    // Define distributions for each type
    if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<T> dist(-10, 10); // Integer distribution
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A[i * n + j] = dist(gen);
                B[i * n + j] = dist(gen);
            }
        }
    } else if constexpr (std::is_floating_point<T>::value) {
        if constexpr (std::is_same<T, float>::value) {
            std::uniform_real_distribution<float> dist(-10.0f, 10.0f); // Float distribution
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    A[i * n + j] = dist(gen);
                    B[i * n + j] = dist(gen);
                }
            }
        } else if constexpr (std::is_same<T, double>::value) {
            std::uniform_real_distribution<double> dist(-10.0, 10.0); // Double distribution
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    A[i * n + j] = dist(gen);
                    B[i * n + j] = dist(gen);
                }
            }
        }
    }
}

template <typename T>
void mmul_ref(const T* A, const T* B, T* C, const int n){
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
            C[i*n+j] = 0.;
            for(int k=0;k<n;++k){
                C[i*n+j]+=A[i*n+k]*B[k*n+j];
            }
        }
    }
}
template <typename T>
void MatrixMaxDifference(const T* A, const T* B,const int n)
{
    T result = 0.;
    for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
        result = std::max( result, std::abs( A[i*n+j] - B[i*n+j] ) );
    std::cout << "Discrepancy between two methods : " << result << std::endl;
}

int main(int argc, char* argv[]){
    if(argc != 3){
        std::cerr << "Usage: " << argv[0] << "<number>\n";
        return 1;
    }
    int n = std::atoi(argv[1]);
    int block_dim = static_cast<int>(std::sqrt(std::atoi(argv[2])));
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    float ms;
    int *A1 = new int[n*n];
    int *B1 = new int[n*n];
    int *C1 = new int[n*n];
    //int *Cref = new int[n*n];
    fillmatrix(A1,B1,n);
    cudaEventRecord(start, 0);
    matmul_1(A1,B1,C1,n,block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms,start,stop);
    std::cout<<C1[0]<<std::endl;
    std::cout<<C1[n*n-1]<<std::endl;
    std::cout<<"Total Time: "<<ms<<" ms\n";
    
    /*mmul_ref(A1,B1,Cref,n);
    MatrixMaxDifference(C1,Cref,n);
    std::cout<<Cref[0]<<std::endl;
    std::cout<<Cref[n*n-1]<<std::endl;
    */

    float *A2 = new float[n*n];
    float *B2 = new float[n*n];
    float *C2 = new float[n*n];
    //double *Cref = new double[n*n];
    fillmatrix(A2,B2,n);
    cudaEventRecord(start, 0);
    matmul_2(A2,B2,C2,n,block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms,start,stop);
    std::cout<<C2[0]<<std::endl;
    std::cout<<C2[n*n-1]<<std::endl;
    std::cout<<"Total Time: "<<ms<<" ms\n";
    
    /*mmul_ref(A2,B2,Cref,n);
    MatrixMaxDifference(C2,Cref,n);
    std::cout<<Cref[0]<<std::endl;
    std::cout<<Cref[n*n-1]<<std::endl;
    */

    double *A3 = new double[n*n];
    double *B3 = new double[n*n];
    double *C3 = new double[n*n];
    //double *Cref = new double[n*n];
    fillmatrix(A3,B3,n);
    std::fill(C3, C3 + n * n, 0.0);
    cudaEventRecord(start, 0);
    matmul_3(A3,B3,C3,n,block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms,start,stop);
    std::cout<<C3[0]<<std::endl;
    std::cout<<C3[n*n-1]<<std::endl;
    std::cout<<"Total Time: "<<ms<<" ms\n";
    /*mmul_ref(A3,B3,Cref,n);
    MatrixMaxDifference(C3,Cref,n);
    std::cout<<Cref[0]<<std::endl;
    std::cout<<Cref[n*n-1]<<std::endl;
    */
    delete[] A1;
    delete[] A2;
    delete[] A3;
    delete[] B1;
    delete[] B2;
    delete[] B3;
    delete[] C1;
    delete[] C2;
    delete[] C3;
    //delete[] Cref;
    return 0;
}
