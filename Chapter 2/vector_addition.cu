#include "vector_addition.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("%s in %s at line %d\n", \
            cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) 
        C[idx] = A[idx] + B[idx];
}


void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    // Part 1: Allocate device memory for A, B, and C
    // Copy A and B to device memory
    CUDA_CHECK(cudaMalloc((void**)&A_d, size));
    CUDA_CHECK(cudaMalloc((void**)&B_d, size));
    CUDA_CHECK(cudaMalloc((void**)&C_d, size));

    CUDA_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));
    

    // Part 2: Call kernel – to launch a grid of threads
    // to perform the actual vector addition
    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);


    // Part 3: Copy C from the device memory
    // Free device vectors

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}