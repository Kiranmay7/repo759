#include <iostream>
#include "matmul.h"
#include <random>
#include <chrono>
#include <ratio>
#include "omp.h"
using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

void reset(double *C,int n){
    for(int i=0;i<n*n;++i){
        C[i] = 0;
    }
}
void print(char ch,double *A,int n){
    //std::cout<<"\n"<<ch<<" Matrix\n";
    for(int i=0;i<n;++i)
    {   for(int j=0;j<n;++j){
            std::cout<<A[i*n+j]<<" ";
        }
        std::cout<<"\n";
    }
}
void print(char ch,const std::vector<double>& A,int n){
    //std::cout<<"\n"<<ch<<" Matrix\n";
    for(int i=0;i<n;++i)
    {   for(int j=0;j<n;++j){
            std::cout<<A[i*n+j]<<" ";
        }
        std::cout<<"\n";
    }
}
void print(int n,double *A,high_resolution_clock::time_point s,high_resolution_clock::time_point e){
    std::cout<<n<<"\n"<<A[n*n-1]<<"\n";
    // Convert the calculated duration to a double using the standard library
    duration<double, std::milli> duration_sec;
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(e - s);
    // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
    cout << "Total time: " << duration_sec.count() << "ms\n";
}
int main(int argc, char* argv[]) {
    /*Command Line input*/
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <number>";
        return 1;
    }
    int n = std::atoi(argv[1]);
    int t = std::atoi(argv[2]);
    std::random_device rd; // Used to seed the generator
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist1(-10.0, 10.0); // Generate numbers between -10.0 and 10.0
    std::uniform_real_distribution<float> dist2(-10.0, 10.0); // Generate numbers between -10.0 and 10.0

    /*Start CPU Timer*/
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    // Get the starting timestamp
    //start = high_resolution_clock::now();

    double A[n*n];// = {1,1,1,1,1,1,1,1,1};
    double B[n*n];// = {1,1,1,1,1,1,1,1,1};
    std::vector<double> A1(n * n);
    std::vector<double> B1(n * n);
    double *C = new double[n*n];
    for(int i = 0; i<n;++i)
    {   for(int j=0;j<n;++j){
            A[i*n+j] = dist1(gen);
            B[i*n+j] = dist2(gen);
            A1[i*n+j] = A[i*n+j];
            B1[i*n+j] = B[i*n+j];
        }
    }
    reset(C,n);
    start = high_resolution_clock::now();
    omp_set_num_threads(t);
    mmul2(A,B,C,n);
    end = high_resolution_clock::now();
    //std::cout<<"\nMatMul 2 output Matrix";
    //print('C',C,n);
    print(n,C,start,end);

    reset(C,n);

    delete[] C;
    return 0;
}