#include "msort.h"
#include <iostream>
#include "omp.h"

int compare(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}
void merge(int *arr, int left, int mid,int right){
    int L_arr_size,R_arr_size,i,j,k;
    L_arr_size = mid - left + 1;
    R_arr_size = right - mid;

    int* Left_arr = new int[L_arr_size];
    int* Right_arr = new int[R_arr_size];
    //#pragma omp parallel for
    for (i = 0; i < L_arr_size; i++)
    {
        Left_arr[i] = arr[left + i];
    }
    //#pragma omp parallel for
    for (j = 0; j < R_arr_size ; j++)
    {
        Right_arr[j] = arr[mid + 1 + j];
    }
    //#pragma omp parallel
    {
        for(i=0,j=0,k=left;i < L_arr_size && j < R_arr_size;++k){
            if(Left_arr[i]<=Right_arr[j]){
                arr[k] = Left_arr[i];
                ++i;
            }
            else{
                arr[k] = Right_arr[j];
                ++j;
            }
        }

        for(;i<L_arr_size;++i,++k){
            arr[k] = Left_arr[i];
        }
        for(;j<R_arr_size;++j,++k){
            arr[k] = Right_arr[j];
        }
    }
}

void mergeSort(int *arr, int left, int right, int th) {
    if(right-left+1<=th){
        std::qsort(&arr[left],right-left+1,sizeof(int),compare);
    }
    else if (left < right) {
        int mid = left + (right-left) / 2;
        //std::cout<<"mid = "<<mid;
        #pragma omp task shared(arr)
        mergeSort(arr, left, mid,th);
        #pragma omp task shared(arr)
        mergeSort(arr, mid + 1, right,th);
        #pragma omp taskwait
        merge(arr, left, mid, right);
    }
    else{return;}
}

void msort(int* arr, const std::size_t n, const std::size_t threshold){
    #pragma omp parallel
    {
        #pragma omp single
        mergeSort(arr, 0, n,threshold);

    }
}