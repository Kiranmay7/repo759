#include <iostream>
#include "msort.h"
#include <random>
#include <chrono>
#include <ratio>
#include "omp.h"
using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char* argv[]) // int argc, char* argv[]
{
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <number>\n";
        return 1;
    }
    int n = std::atoi(argv[1]);
    int t = std::atoi(argv[2]);
    int ts = std::atoi(argv[3]);
    int arr[n];// = { 12, 11, 13, 5, 6, 7 };
    //int n = 6;
    std::random_device rd; // Used to seed the generator
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist1(-1000, 1000); // Generate numbers between -1000 and 1000
    //std::cout << "Given vector is";
    for (int i = 0; i < n; i++){
        arr[i] = dist1(gen);
    }
    /*Start CPU Timer*/
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    //Setting OMP threads
    omp_set_num_threads(t);
    // Get the starting timestamp
    start = high_resolution_clock::now();
    msort(arr, n - 1, ts);
    end = high_resolution_clock::now();
    //std::cout << "\nSorted vector is \n";
    std::cout << arr[0] << std::endl << arr[n-1] << std::endl;
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << "Total Time: " << duration_sec.count() << "ms\n";
    return 0;
}