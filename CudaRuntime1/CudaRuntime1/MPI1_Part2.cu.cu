
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <math.h>

void initializeMatrix(float* Array, int SIZE) {
    // Function associates a random float at each index of the matrix
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            Array[i * SIZE + j] = (float)(rand() / RAND_MAX);

        }
    }
}

void transferFunction(int SIZE) {
    // Initialize byte size by matrix size
    int BYTES = SIZE * SIZE * sizeof(float);
       
    // Initialize and allocate memory to host Matrix A and B
    float* hostMatrixA = 0;
    float* hostMatrixB = 0;
    cudaMallocHost((void**)&hostMatrixA, BYTES);
    cudaMallocHost((void**)&hostMatrixB, BYTES);
    initializeMatrix(hostMatrixA, SIZE);
    initializeMatrix(hostMatrixB, SIZE);

    // Initialize and allocate memory to device Matrix A and B
    float* deviceMatrixA = 0;
    float* deviceMatrixB = 0;
    cudaMalloc((void**)&deviceMatrixA, BYTES);
    cudaMalloc((void**)&deviceMatrixB, BYTES);

    // Initialize host to device and device to host times
    float HTD = 0;
    float DTH = 0;
    
    // CUDA event variables
    cudaEvent_t startTime, stopTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&stopTime);
    cudaDeviceSynchronize();

    // Start recording and copy the host matrices from the host to the device
    cudaEventRecord(startTime, 0);
    cudaMemcpy(deviceMatrixA, hostMatrixA, BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, hostMatrixB, BYTES, cudaMemcpyHostToDevice);

    // Stop the timer and associate it with host to device variable
    cudaEventRecord(stopTime);
    cudaEventSynchronize(stopTime);
    cudaEventElapsedTime(&HTD, startTime, stopTime);

    // Start recording and copy the device matrices from device to host
    cudaEventRecord(startTime, 0);
    cudaMemcpy(hostMatrixA, deviceMatrixA, BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostMatrixB, deviceMatrixB, BYTES, cudaMemcpyDeviceToHost);

    // Stop the timer and associate it with device to host variable
    cudaEventRecord(stopTime); 
    cudaEventSynchronize(stopTime);
    cudaEventElapsedTime(&DTH, startTime, stopTime);

    printf("Host to Device for %d matrix size: % .2fms\n", SIZE, HTD);
    printf("Device to Host for %d matrix size: %.2fms\n", SIZE, DTH);

    // Free allocated memory and reset time events 
    cudaEventDestroy(startTime);
    cudaEventDestroy(stopTime);
    cudaFreeHost(hostMatrixA);
    cudaFreeHost(hostMatrixB);
    cudaFree(deviceMatrixA);
    cudaFree(deviceMatrixB);
    cudaDeviceReset();
}

void matrixMul(float* resultMatrix, float* MatrixA, float* MatrixB, int SIZE) {
    
    // Initialize clock 
    clock_t startTime, stopTime;
    startTime = clock();

    // Matrix multiplication algorithm
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            resultMatrix[i * SIZE + j] = 0;

            for (int k = 0; k < SIZE; k++) {
                resultMatrix[i * SIZE + j] += MatrixA[i * SIZE + k] * MatrixB[k * SIZE + j];
            }
        }
    }

    // Stop the time and print results
    stopTime = clock();
    printf("CPU Matrix Multiplication % .2fms\n", (double)stopTime - startTime);

}
__global__ void kernelMatrixMul(float* Result, float* MatrixA, float* MatrixB, int SIZE) {

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            float matrixSum = 0;
            for (int k = 0; k < SIZE; k++) {
                matrixSum += MatrixA[i * SIZE + k] * MatrixB[k * SIZE + j];
            }
            Result[i * SIZE + j] = matrixSum;
        }
    }


    return;
}


void verifyMatrix(float* CPUMatrix, float* GPUMatrix, int SIZE) {

    // For every matrix index, check if the CPU and GPU results match within an allowance of 0.01
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (fabs(CPUMatrix[i * SIZE + j] - GPUMatrix[i * SIZE + j]) > 0.01) {
                printf("TEST FAILED\n");
                return;
            }
        }
    }
    // Print passed if matrices match
    printf("TEST PASSED\n");
    return;

}

void gpuMatrixMul(float* Result, float* MatrixA, float* MatrixB, int SIZE) {
    // Determine byte size of matrix 
    int BYTES = SIZE * SIZE * sizeof(float);

    // Define device matrices and allocate them memory
    float* deviceMatrixA;
    float* deviceMatrixB;
    float* deviceResultMatrix;

    cudaMalloc(&deviceMatrixA, BYTES);
    cudaMalloc(&deviceMatrixB, BYTES);
    cudaMalloc(&deviceResultMatrix, BYTES);

    // Copy the matrices from the host to the device
    cudaMemcpy(deviceMatrixA, MatrixA, BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, MatrixB, BYTES, cudaMemcpyHostToDevice);

    // Establish time and CUDA events
    float time = 0;

    cudaEvent_t startTime, stopTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&stopTime);
    cudaDeviceSynchronize();

    // Make grid and block dimensions 1 
    dim3 threadsPerBlock(1, 1);
    dim3 blocksPerGrid(1, 1);

    // Start recording and start matrix multiplication
    cudaEventRecord(startTime, 0);
    kernelMatrixMul << <threadsPerBlock, blocksPerGrid >> > (deviceResultMatrix, deviceMatrixA, deviceMatrixB, SIZE);

    // Stop recording and store the time result in time variable
    cudaEventRecord(stopTime, 0);
    cudaEventSynchronize(stopTime);
    cudaEventElapsedTime(&time, startTime, stopTime);

    // Copy device matrix to host
    cudaMemcpy(Result, deviceResultMatrix, BYTES, cudaMemcpyDeviceToHost);

    // Print results
    printf("GPU Matrix Multiplication Time: %.2f\n", time);

    // Free event and memory
    cudaEventDestroy(startTime);
    cudaEventDestroy(stopTime);
    cudaFree(deviceMatrixA);
    cudaFree(deviceMatrixB);
    cudaFree(deviceResultMatrix);
    cudaDeviceReset();
}

__global__ void kernelMultipleMatrixMul(float* Result, float* MatrixA, float* MatrixB, int SIZE) {
    // Calculate thread row and columns 
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    // Matrix multiplication with threads
    if (Row < SIZE && Col < SIZE)
    {
        float sum = 0;
        for (int k = 0; k < SIZE; ++k)
            sum += MatrixA[Row * SIZE + k] * MatrixB[k * SIZE + Col];
        Result[Row * SIZE + Col] = sum;
    }

    return;
}

void gpuThreadMatrixMul(float* Result, float* MatrixA, float* MatrixB, int SIZE, int blockWidth) {
    // Determine byte size
    int BYTES = SIZE * SIZE * sizeof(float);

    // Initialize and allocate memory to device matrices
    float* deviceMatrixA;
    float* deviceMatrixB;
    float* deviceResultMatrix;

    cudaMalloc(&deviceMatrixA, BYTES);
    cudaMalloc(&deviceMatrixB, BYTES);
    cudaMalloc(&deviceResultMatrix, BYTES);

    // Copy host matrices to the device 
    cudaMemcpy(deviceMatrixA, MatrixA, BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, MatrixB, BYTES, cudaMemcpyHostToDevice);

    // Initialize time and CUDA events
    float time = 0;

    cudaEvent_t startTime, stopTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&stopTime);
    cudaDeviceSynchronize();

    // Determine factors for grid and block dimenstions
    int NumBlocks = SIZE / blockWidth;
    if (SIZE % blockWidth) NumBlocks++;

    dim3 dimGrid(NumBlocks, NumBlocks);
    dim3 dimBlock(blockWidth, blockWidth);
    
    // Start recording and call the thread multiplication
    cudaEventRecord(startTime, 0);
    kernelMultipleMatrixMul << <dimGrid, dimBlock >> > (deviceResultMatrix, deviceMatrixA, deviceMatrixB, SIZE);
    // Stop recording and store time in appropriate variable
    cudaEventRecord(stopTime, 0);
    cudaEventSynchronize(stopTime);
    cudaEventElapsedTime(&time, startTime, stopTime); 

    // Copy the finalized matrix over to the host
    cudaMemcpy(Result, deviceResultMatrix, BYTES, cudaMemcpyDeviceToHost);

    // Print results
    printf("GPU Matrix Multiplication Time for %d size and %d block width: %.2f\n", SIZE, blockWidth, time);

    // Free memory and events
    cudaEventDestroy(startTime);
    cudaEventDestroy(stopTime);
    cudaFree(deviceMatrixA);
    cudaFree(deviceMatrixB);
    cudaFree(deviceResultMatrix);
    cudaDeviceReset();
}

int main()
{
    //// Number of devices
    //int nd;

    //// Name of Devices
    //char name[50];

    //// Print Number of Cuda Devices
    //cudaGetDeviceCount(&nd);
    //printf("Number of CUDA devices: %d\n", nd);

    //// Gather device information for each identified CUDA device 
    //for (int d = 0; d < nd; d++)
    //{
    //    // Struct with device information
    //    cudaDeviceProp dp;
    //    cudaGetDeviceProperties(&dp, d);

    //    // Device Type
    //    printf("Device Type: %s\n", dp.name);
    //    //Clock Rate
    //    printf("Clock Rate: %d\n", dp.clockRate);

    //    // Number of streaming multiprocessors
    //    printf("Number of Streaming Multiprocessors: %d\n", dp.multiProcessorCount);

    //    // Number of cores
    //    printf("Number of Cores: %d\n", 128 * dp.multiProcessorCount);

    //    // Warp size
    //    printf("Warp Size: %d\n", dp.warpSize);

    //    // Global, constant, and shared memory
    //    printf("Global Memory: %zu\n", dp.totalGlobalMem);
    //    printf("Amount of Constant Memory: %zu\n", dp.totalConstMem);
    //    printf("Amount of Shared Memory per Block: %zu\n", dp.sharedMemPerBlock);

    //    // Reg and threads per block
    //    printf("Number of Registers Available Per Block: %d\n", dp.regsPerBlock);
    //    printf("Maximum Number of Threads Per Block: %d\n", dp.maxThreadsPerBlock);

    //    // Dimension of block and grid
    //    printf("Maximum Size of each Dimension of a Block: %dx%dx%d\n", dp.maxThreadsDim[0], dp.maxThreadsDim[1], dp.maxThreadsDim[2]);
    //    printf("Maximum Size of each Dimension of a Grid: %dx%dx%d\n", dp.maxGridSize[0], dp.maxGridSize[1], dp.maxGridSize[2]);

    //}

    //transferFunction(256);
    //transferFunction(512);
    //transferFunction(1024);
    //transferFunction(2048);
    //transferFunction(4096);
    
  // Initialize matrices
  /*float* MatrixA;
  float* MatrixB;
  float* resultMatrix1;
  float* resultMatrix2;*/

  // Initialize to 256
  //int BYTES = 0;

  //BYTES = 256 * 256 * sizeof(float);

  //MatrixA = (float*)malloc(BYTES);
  //MatrixB = (float*)malloc(BYTES);
  //resultMatrix1 = (float*)malloc(BYTES);
  //resultMatrix2 = (float*)malloc(BYTES);

  //int sizes[5] = {256, 512, 1024};

  ////For the three required matrix sizes
  //for(int i = 0; i < 5; i++){
  //    BYTES = sizes[i] * sizes[i] * sizeof(float);

  //    // allocated the necessary memory
  //    MatrixA = (float*)realloc(MatrixA, BYTES);
  //    MatrixB = (float*)realloc(MatrixB, BYTES);
  //    resultMatrix1 = (float*)realloc(resultMatrix1, BYTES);
  //    resultMatrix2 = (float*)realloc(resultMatrix2, BYTES);

  //    // Reinitialize larger arrays
  //    initializeMatrix(MatrixA, sizes[i]);
  //    initializeMatrix(MatrixB, sizes[i]);

  //    // Call CPU multiplication
  //    matrixMul(resultMatrix2, MatrixA, MatrixB, sizes[i]);
  //    // Call GPU multiplication
  //    gpuMatrixMul(resultMatrix1, MatrixA, MatrixB, sizes[i]);
  //    // Verify they are the same value
  //    verifyMatrix(resultMatrix1, resultMatrix2, sizes[i]);

  //}

  //return 0;

  // Initialize all matrices
  float* MatrixA;
  float* MatrixB;
  float* resultMatrix1;
  float* resultMatrix2;

  // Set matrices to first memory size
  int BYTES = 0;

  BYTES = 256 * 256 * sizeof(float);

  MatrixA = (float*)malloc(BYTES);
  MatrixB = (float*)malloc(BYTES);
  resultMatrix1 = (float*)malloc(BYTES);
  resultMatrix2 = (float*)malloc(BYTES);

  // Initialize matrices
  initializeMatrix(MatrixA, 256);
  initializeMatrix(MatrixB, 256);

  // Thread block sizes/ matrix sizes
  int list[5] = { 2, 4, 8, 16, 32 };
  int sizes[5] = {256, 512, 1024, 2048, 4096};

  // Loop through sizes
  for(int i = 0; i < 5; i++){
      BYTES = sizes[i] * sizes[i] * sizeof(float);

      // Allocate memory for new size
      MatrixA = (float*)realloc(MatrixA, BYTES);
      MatrixB = (float*)realloc(MatrixB, BYTES);
      resultMatrix1 = (float*)realloc(resultMatrix1, BYTES);
      resultMatrix2 = (float*)realloc(resultMatrix2, BYTES);

      // Initialize larger matrices
      initializeMatrix(MatrixA, sizes[i]);
      initializeMatrix(MatrixB, sizes[i]);

      for (int j = 0; j < 5; j++) {
          // Call GPU matrix multiplications with differeing threads
          gpuThreadMatrixMul(resultMatrix1, MatrixA, MatrixB, sizes[i], list[j]);
          // Call CPU matrix multiplication
          matrixMul(resultMatrix2, MatrixA, MatrixB, sizes[i]);
          // Ensure the multiplcation is the same result
          verifyMatrix(resultMatrix1, resultMatrix2, sizes[i]);
      }
  }

  return 0;
}


