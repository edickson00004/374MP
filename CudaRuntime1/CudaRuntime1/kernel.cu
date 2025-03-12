#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

/*
__global__ void GPU_matrix_multiply(float* M, float* N, float* P, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0;
    for (int i = 0; i < n; i++) {
     temp += M[row * n + i] * N[i * n + col];
    }
    P[row * n + col] = temp;
}
*/
__global__ void GPU_matrix_multiply(float* M, float* N, float* P, int n) {

    for (int i = 0; i < n; i++) {
        float temp = 0;
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                temp += M[i * n + k] * N[k * n + j];
            }
            P[i * n + j] = temp;
        }
    }
}


void gpu_Multi(float* M, float* N, float* P, int n) {
    size_t size = n * n * sizeof(float);
    float* d_M;
    float* d_N;
    float* d_P;
    cudaMalloc((void**)&d_P, size);
    cudaMalloc((void**)&d_M, size);
    cudaMalloc((void**)&d_N, size);

    cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, size, cudaMemcpyHostToDevice);
    float time = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    
    dim3 threads(1, 1);
    dim3 blocks(1, 1);
    cudaEventRecord(start, 0);
    GPU_matrix_multiply << < blocks, threads >> > (d_M, d_N, d_P, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost);
    printf("GPU time: %f\n", time);

}

void cpu_Mulit(float* M, float* N, float* P, int n) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);
    float time;
    for (int i = 0; i < n; i++) {
        float temp = 0;
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                temp += M[i * n + k] * N[k * n + j];
            }
            P[i * n + j] = temp;
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("CPU time: %f\n", time);
}

void transferTime(float* M, float* N, int size) {
    float* d_M;
    float* d_N;
    float deviceTime, hostTime;
    cudaMalloc((void**)&d_M, size);
    cudaMalloc((void**)&d_N, size);
    cudaEvent_t start, stop;

    //From host to device
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);
    cudaMemcpyAsync(d_M, M, size, cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_N, N, size, cudaMemcpyHostToDevice, 0);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&deviceTime, start, stop);

    //from device to host
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);
    cudaMemcpyAsync(M, d_M, size, cudaMemcpyDeviceToHost, 0);
    cudaMemcpyAsync(N, d_N, size, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&hostTime, start, stop);

    printf("Host to Device time: %f\n", deviceTime);
    printf("Device to Host time: %f\n", hostTime);

}

void genRandMatrix(float* matrix, int n) {
    int limit = n * n;
    for (int i = 0; i < limit; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}
int main()
{
    int n = 256;
    size_t size = n * n * sizeof(float);
    float* M = (float*)malloc(size);
    float* N = (float*)malloc(size);
    float* P = (float*)malloc(size);

    genRandMatrix(M, n);
    genRandMatrix(N, n);

    /* Part 2.1
        for (int i = 0; i < 10; i++) {
        printf("Trial %d -----------------------------------\n", i + 1);
        genRandMatrix(M, n);
        genRandMatrix(N, n);
        transferTime(M, N, size);
    }
    */
    //Part 2.2
    gpu_Multi(M, N, P, n);
    cpu_Mulit(M, N, P, n);



    return 0;
}