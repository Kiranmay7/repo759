#include "convolution.h"
#include <iostream>
void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m){
    #pragma omp parallel for
    for(long unsigned int x = 0;x<n;++x){
        for(long unsigned int y = 0;y<n;++y){
            output[x*n+y] = 0;
            for(long unsigned int i=0;i<m;++i){
                for(long unsigned int j=0;j<m;++j){
                    long unsigned int k = x+i-((m-1)/2);
                    long unsigned int l = y+j-((m-1)/2);
                    if(!(k>=0 && k<n)){
                        if(l>=0 && l<n){
                            output[x*n+y]+= mask[i*m+j];
                        }
                    }
                    else if(!(l>=0 && l<n)){
                        if(k>=0 && k<n){
                            output[x*n+y]+= mask[i*m+j];
                        }
                    }
                    else if((k>=0&&k<n)&&(l>=0&&l<n)){
                        output[x*n+y]+= mask[i*m+j]*image[k*n+l];
                    }
                }
            }
        }
    }
}