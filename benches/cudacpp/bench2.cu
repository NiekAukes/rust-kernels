#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void matrix_mul(const int* a, const int* b, int* c, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int i = t / n;
    int j = t % n;
    
    if (i < n && j < n) {
        int sum = 0;
        for (int k = 0; k < n; ++k) {
            sum += a[i * n + k] * b[k * n + j];
        }
        c[i * n + j] = sum;
    }
}

int main() {
    int n = 1024;
    size_t size = n * n * sizeof(int);

    std::vector<int> a(n * n), b(n * n), c(n * n, 0);
    for (int i = 0; i < n * n; ++i) {
        //a[i] = (i * 1234567) % 1000;
        //b[i] = (i * 7654321) % 1000;
        a[i] = i;
        b[i] = i;
    }

    int *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));
    CHECK_CUDA(cudaMalloc(&d_c, size));

    CHECK_CUDA(cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice));

    int threads_per_block = 64;
    int blocks = (n * n + threads_per_block - 1) / threads_per_block;

    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 100; ++iter) {
        matrix_mul<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Matrix size: " << n << "x" << n << " - GPU time: " << duration << " ms" << std::endl;

    std::cout << "First 10 elements of the result:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return 0;
}

