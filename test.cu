#include <iostream>
#include <cuda_runtime.h>

int main() {
    int         count = 0;
    cudaError_t err;

    err = cudaGetDeviceCount(&count);
    std::cout << "cudaGetDeviceCount: " << (int)err << " (" << cudaGetErrorString(err)
              << "), count = " << count << std::endl;

    if (err != cudaSuccess || count == 0) {
        std::cout << "No usable CUDA device." << std::endl;
        return 0;
    }

    err = cudaSetDevice(0);
    std::cout << "cudaSetDevice(0): " << (int)err << " (" << cudaGetErrorString(err) << ")"
              << std::endl;

    float* p = nullptr;
    err      = cudaMalloc(&p, 4);
    std::cout << "cudaMalloc: " << (int)err << " (" << cudaGetErrorString(err) << ")" << std::endl;

    if (err == cudaSuccess)
        cudaFree(p);
    return 0;
}
