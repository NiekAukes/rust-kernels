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

void matrix_mul_cpu(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int sum = 0;
            for (int k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

const int ns[] = { 4, 16, 64, 256, 512, 1024, 2048, 4096, 8192 };

int main() {
    for (int s = 0; s < 9; ++s) {
        int n = ns[s];
        size_t size = n * n * sizeof(int);

        std::cout << "Matrix size: " << n << "x" << n << std::endl;

        std::vector<int> a(n * n), b(n * n), c(n * n, 0), c_cpu(n * n, 0);
        
        for (int i = 0; i < n * n; ++i) {
            a[i] = (i * 1234567) % 1000;
            b[i] = (i * 7654321) % 1000;
        }

        auto gpu_start = std::chrono::high_resolution_clock::now();
        int *d_a, *d_b, *d_c;
        CHECK_CUDA(cudaMalloc(&d_a, size));
        CHECK_CUDA(cudaMalloc(&d_b, size));
        CHECK_CUDA(cudaMalloc(&d_c, size));

        CHECK_CUDA(cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice));

        int threads_per_block = 64;
        int blocks = (n * n + threads_per_block - 1) / threads_per_block;

    
        matrix_mul<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost));

        auto gpu_end = std::chrono::high_resolution_clock::now();
        auto gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start).count();

        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;

        CHECK_CUDA(cudaFree(d_a));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_c));
    }
    return 0;
}
