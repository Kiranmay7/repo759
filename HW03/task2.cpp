#include <iostream>
#include "convolution.h"
#include <random>
#include <chrono>
#include <ratio>
#include "omp.h"
using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char* argv[]) {
    /*Command Line input*/
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <number>\n";
        return 1;
    }
    int n = std::atoi(argv[1]);
    int t = std::atoi(argv[2]);
    std::random_device rd; // Used to seed the generator
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist1(-10.0, 10.0); // Generate numbers between -10.0 and 10.0
    std::uniform_real_distribution<float> dist2(-1.0, 1.0); // Generate numbers between -1.0 and 1.0
    /*Start CPU Timer*/
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    // Get the starting timestamp
    start = high_resolution_clock::now();
    int m = 3;
    float image[n*n];
    float mask[m*m];
    for(int i = 0; i<n;++i)
    {   for(int j=0;j<n;++j){
            image[i*n+j] = dist1(gen);
        }
    }
    /*std::cout<<"Image Matrix\n";
    for(int i=0;i<n;++i)
    {   for(int j=0;j<n;++j){
            std::cout<<image[i*n+j]<<" ";
        }
        std::cout<<"\n";
    }*/
    //std::cout<<"Mask Matrix\n";
    for(int i = 0; i<m;++i)
    {   for(int j=0;j<m;++j){
            mask[i*m+j] = dist2(gen);
        }
    }
    /*for(int i=0;i<m;++i)
    {   for(int j=0;j<m;++j){
            std::cout<<mask[i*m+j]<<" ";
        }
        std::cout<<"\n";
    }*/
    float *output = new float[n*n];
    omp_set_num_threads(t);
    convolve(image, output, n, mask, m);
    // Get the ending timestamp
    end = high_resolution_clock::now();

    std::cout<<output[0]<<"\n"<<output[n*n-1];


    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
    cout << "\nTotal time: " << duration_sec.count() << "ms\n";

    delete[] output;
    return 0;
}