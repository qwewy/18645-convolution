#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <math.h>
#include "sobel.hpp"

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

/* assign thread per pixel, and compute individually */
__global__ void convolveBasic(unsigned char *image,
                              int *convResult,
                              int *convMax,
                              int *convMin,
                              char *kernelX,
                              char *kernelY,
                              int W,
                              int H) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int totalX = 0;
    int totalY = 0;

    if (imageX >= 0 && imageX < W && imageY >= 0 && imageY < H) {
        for (int i = 0; i < KERNEL_SIZE; i ++) { // TODO: Use macro
            for (int j = 0; j < KERNEL_SIZE; j ++) {
                int currRow = imageY + i - 1;
                int currCol = imageX + j - 1;
                if (currRow >= 0 &&
                    currRow < H &&
                    currCol >= 0 &&
                    currCol < W) {
                    totalX += (int)(unsigned int)image[currRow * W + currCol] *
                              (int)kernelX[i * KERNEL_SIZE + j];
                    totalY += (int)(unsigned int)image[currRow * W + currCol] *
                              (int)kernelY[i * KERNEL_SIZE + j];
                }
            }
        }
        int currConvResult = sqrtf(totalX * totalX + totalY * totalY);
        convResult[imageY * W + imageX] = currConvResult;

        atomicMax(convMax, currConvResult);
        atomicMin(convMin, currConvResult);
    }
}

__global__ void normalize(int *convResult,
                          unsigned char *outImage,
                          int *convMax,
                          int *convMin,
                          int W,
                          int H) {
    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    if (imageX >= 0 && imageX < W && imageY >= 0 && imageY < H) {
        double resultNorm =
            (double)(convResult[imageY * W + imageX] - *convMin) /
            (double)(*convMax - *convMin);
        outImage[imageY * W + imageX] = MAX_BRIGHTNESS * resultNorm;
    }
}

void sobelPar(unsigned char *inImage,
              unsigned char *outImage,
              const char *kernelX,
              const char *kernelY,
              int W,
              int H) {

    unsigned char *deviceImage;
    int *deviceConvResult;
    char *deviceKernelX, *deviceKernelY;
    int *convMax, *convMin;

    int initConvMax = INT_MIN;
    int initConvMin = INT_MAX;
    unsigned int imageSize = W * H * sizeof(unsigned char);
    unsigned int kernelSize = KERNEL_SIZE * KERNEL_SIZE * sizeof(char);
    unsigned int convResultSize = W * H * sizeof(int);

    cudaMalloc(&deviceImage, imageSize);
    cudaMalloc(&deviceKernelX, kernelSize);
    cudaMalloc(&deviceKernelY, kernelSize);
    cudaMalloc(&deviceConvResult, convResultSize);
    cudaMalloc(&convMax, sizeof(int));
    cudaMalloc(&convMin, sizeof(int));
    cudaMemcpy(deviceImage, inImage, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceKernelX, kernelX, kernelSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceKernelY, kernelY, kernelSize, cudaMemcpyHostToDevice);
    cudaMemcpy(convMax, &initConvMax, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(convMin, &initConvMin, sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);
    size_t gridWidth = (W + blockDim.x - 1) / blockDim.x;
    size_t gridHeight = (H + blockDim.y - 1) / blockDim.y;
    dim3 gridDim(gridWidth, gridHeight);

    // convolution
    convolveBasic<<<gridDim, blockDim>>>(deviceImage,
                                         deviceConvResult,
                                         convMax,
                                         convMin,
                                         deviceKernelX,
                                         deviceKernelY,
                                         W,
                                         H);
    cudaDeviceSynchronize();

    // normalize results
    normalize<<<gridDim, blockDim>>>(deviceConvResult,
                                     deviceImage,
                                     convMax,
                                     convMin,
                                     W,
                                     H);
    cudaDeviceSynchronize();

    cudaMemcpy(outImage, deviceImage, imageSize, cudaMemcpyDeviceToHost);
}
