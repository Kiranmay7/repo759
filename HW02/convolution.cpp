#include "convolution.h"
#include <iostream>
void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m){       
    for(int x = 0;x<n;++x){
        for(int y = 0;y<n;++y){
            output[x*n+y] = 0;
            for(int i=0;i<m;++i){
                for(int j=0;j<m;++j){
                    int k = x+i-((m-1)/2);
                    int l = y+j-((m-1)/2);
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