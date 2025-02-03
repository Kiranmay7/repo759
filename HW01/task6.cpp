#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <number>\n";
        return 1;
    }
    int num = std::atoi(argv[1]);
    for (int i = 0; i <= num; ++i) {
        //std::cout << i << " ";
        printf("%d ",i);
    }
    std::cout << std::endl;
    for (int i = num; i >= 0; --i) {
        std::cout << i << " ";
    }
    return 0;
}