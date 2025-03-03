#include "matmul.h"
#include <iostream>

void mmul2(const double* A, const double* B, double* C, const unsigned int n){
#pragma omp parallel for
    for(unsigned int i=0;i<n;++i){
        for(unsigned int k=0;k<n;++k){
            for(unsigned int j=0;j<n;++j){
                C[i*n+j]+=A[i*n+k]*B[k*n+j];
            }
        }
    }
}