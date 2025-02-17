#include <iostream>
#include "scan.h"
#include <random>
#include <chrono>
#include <ratio>

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char* argv[]) {
    /*Command Line input*/
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <number>\n";
        return 1;
    }
    long n = std::stol(argv[1]);
    /*Command Line input*/
    std::random_device rd; // Used to seed the generator
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<float> dist(-1.0, 1.0); // Generate numbers between -1.0 and 1.0
    /*Start CPU Timer*/
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    // Get the starting timestamp
    start = high_resolution_clock::now();

    float arr[n];
    for(int i = 0; i<n;++i)
    {
        arr[i] = dist(gen);
    }
    float *output = new float[n];
    scan(arr, output, n);

    // Get the ending timestamp
    end = high_resolution_clock::now();
    
    std::cout<<output[0]<<"\n"<<output[n-1]<<"\n";

    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
    cout << "Total time: " << duration_sec.count() << "ms\n";
    delete[] output;
    return 0;
}