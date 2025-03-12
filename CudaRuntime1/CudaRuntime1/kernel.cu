
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctime>

void transferFunction(int SIZE) {
    int BYTES = SIZE * SIZE * sizeof(float);

    float* hostMatrixA = 0;
    float* hostMatrixB = 0;
    cudaMallocHost((void**)&hostMatrixA, BYTES);
    cudaMallocHost((void**)&hostMatrixB, BYTES);

    float* deviceMatrixA = 0;
    float* deviceMatrixB = 0;

    cudaMalloc((void**)&deviceMatrixA, BYTES);
    cudaMalloc((void**)&deviceMatrixB, BYTES);

    float HTD = 0;
    float DTH = 0;

    cudaEvent_t startTime, stopTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&stopTime);
    cudaDeviceSynchronize();

    cudaEventRecord(startTime, 0);
    cudaMemcpy(deviceMatrixA, hostMatrixA, BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, hostMatrixB, BYTES, cudaMemcpyHostToDevice);
    cudaEventRecord(stopTime); // stop is updated here
    cudaEventSynchronize(stopTime);
    cudaEventElapsedTime(&HTD, startTime, stopTime); //time difference between start and stop

    cudaEventRecord(startTime, 0);
    cudaMemcpy(hostMatrixA, deviceMatrixA, BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostMatrixB, deviceMatrixB, BYTES, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopTime); // stop is updated here
    cudaEventSynchronize(stopTime);
    cudaEventElapsedTime(&DTH, startTime, stopTime);


    printf("Host to Device: % .2fms\n", HTD);
    printf("Device to Host: %.2fms\n", DTH);

    cudaEventDestroy(startTime);
    cudaEventDestroy(stopTime);
    cudaFreeHost(hostMatrixA);
    cudaFreeHost(hostMatrixB);
    cudaFree(deviceMatrixA);
    cudaFree(deviceMatrixB);
    cudaDeviceReset();
}

void initializeMatrix(float* Array, int SIZE) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            Array[i * SIZE + j] = (float)(rand() / RAND_MAX);

        }
    }

}
void matrixMul(int SIZE) {
    int BYTES = SIZE * SIZE * sizeof(float);

    float* MatrixA = (float*)malloc(BYTES);
    float* MatrixB = (float*)malloc(BYTES);
    float* resultMatrix = (float*)malloc(BYTES);

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            MatrixA[i * SIZE + j] = 1;
            MatrixB[i * SIZE + j] = 1;

        }
    }
    clock_t startTime, stopTime;

    startTime = clock();

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            resultMatrix[i * SIZE + j] = 0;

            for (int k = 0; k < SIZE; k++) {
                resultMatrix[i * SIZE + j] += MatrixA[i * SIZE + k] * MatrixB[k * SIZE + j];
            }
        }
    }

    stopTime = clock();

    printf("CPU Matrix Multiplication % .2fms\n", (double)stopTime - startTime);

    free(MatrixA);
    free(MatrixB);
    free(resultMatrix);

}
__global__ void kernelMatrixMul(float* Result, float* MatrixA, float* MatrixB, int SIZE) {

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            Result[i * SIZE + j] = 0;
            for (int k = 0; k < SIZE; k++) {
                Result[i * SIZE + j] += MatrixA[i * SIZE + k] * MatrixB[k * SIZE + j];
            }
        }
    }


    return;
}
void gpuMatrixMul(float* Result, float* MatrixA, float* MatrixB, int SIZE) {
    int BYTES = SIZE * SIZE * sizeof(float);

    float* deviceMatrixA;
    float* deviceMatrixB;
    float* deviceResultMatrix;

    cudaMalloc(&deviceMatrixA, BYTES);
    cudaMalloc(&deviceMatrixB, BYTES);
    cudaMalloc(&deviceResultMatrix, BYTES);

    cudaMemcpy(deviceMatrixA, MatrixA, BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, MatrixB, BYTES, cudaMemcpyHostToDevice);

    float time = 0;

    cudaEvent_t startTime, stopTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&stopTime);
    cudaDeviceSynchronize();

    dim3 threadsPerBlock(1, 1);
    dim3 blocksPerGrid(1, 1);

    cudaEventRecord(startTime, 0);
    kernelMatrixMul << <threadsPerBlock, blocksPerGrid >> > (deviceResultMatrix, deviceMatrixA, deviceMatrixB, SIZE);
    cudaEventRecord(stopTime, 0); // stop is updated here
    cudaEventSynchronize(stopTime);
    cudaEventElapsedTime(&time, startTime, stopTime); //time difference between start and stop

    cudaMemcpy(Result, deviceResultMatrix, BYTES, cudaMemcpyDeviceToHost);

    printf("GPU Matrix Multiplication Time: %.2f", time);
    cudaEventDestroy(startTime);
    cudaEventDestroy(stopTime);
    cudaFree(deviceMatrixA);
    cudaFree(deviceMatrixB);
    cudaFree(deviceResultMatrix);
    cudaDeviceReset();
}

int main()
{
    int nd;
    char name[50];

    cudaGetDeviceCount(&nd);

    printf("Number of CUDA devices: %d\n", nd);
    for (int d = 0; d < nd; d++)
    {
        cudaDeviceProp dp;
        cudaGetDeviceProperties(&dp, d);
        printf("Device Type: %s\n", dp.name);
        printf("Clock Rate: %d\n", dp.clockRate);
        printf("Number of Streaming Multiprocessors: %d\n", dp.multiProcessorCount);
        printf("Number of Cores: %d\n", 128 * dp.multiProcessorCount);
        printf("Warp Size: %d\n", dp.warpSize);
        printf("Global Memory: %zu\n", dp.totalGlobalMem);
        printf("Amount of Constant Memory: %zu\n", dp.totalConstMem);
        printf("Amount of Shared Memory per Block: %zu\n", dp.sharedMemPerBlock);
        printf("Number of Registers Available Per Block: %d\n", dp.regsPerBlock);
        printf("Maximum Number of Threads Per Block: %d\n", dp.maxThreadsPerBlock);
        printf("Maximum Size of each Dimension of a Block: %dx%dx%d\n", dp.maxThreadsDim[0], dp.maxThreadsDim[1], dp.maxThreadsDim[2]);
        printf("Maximum Size of each Dimension of a Grid: %dx%dx%d\n", dp.maxGridSize[0], dp.maxGridSize[1], dp.maxGridSize[2]);

    }

    transferFunction(256);
    transferFunction(512);
    transferFunction(1024);
    transferFunction(2048);
    transferFunction(4096);

    matrixMul(256);
    matrixMul(512);
    matrixMul(1024);

    float* MatrixA = 0;
    float* MatrixB = 0;
    float* resultMatrix = 0;
    int BYTES = 0;

    BYTES = 256 * 256 * sizeof(float);

    MatrixA = (float*)malloc(BYTES);
    MatrixB = (float*)malloc(BYTES);
    resultMatrix = (float*)malloc(BYTES);
    initializeMatrix(MatrixA, 256);
    initializeMatrix(MatrixB, 256);

    gpuMatrixMul(resultMatrix, MatrixA, MatrixB, 256);

    return 0;
}

