#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << " : " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void add_one(int* a) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    *(a+t) = t + 1;
}

int main() {
    // bench the kernel function executing 100 times
    int n = 1024;
    size_t size = n * n * sizeof(int);

    std::vector<int> a(n * n);
    for (int i = 0; i < n * n; ++i) {
        a[i] = i;
    }

    int *d_a;
    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice));

    int threads_per_block = 64;
    int blocks = (n * n + threads_per_block - 1) / threads_per_block;

    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 10000; ++iter) {
        add_one<<<blocks, threads_per_block>>>(d_a);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaMemcpy(a.data(), d_a, size, cudaMemcpyDeviceToHost));
    
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        
    return 0;
}
