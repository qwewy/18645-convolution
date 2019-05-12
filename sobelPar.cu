#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <math.h>
#include <sobel.hpp>
#include <stdio.h>

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32
#define THREADS_PER_BLOCK 1024

__constant__ char cudaKernelX[KERNEL_SIZE * KERNEL_SIZE];
__constant__ char cudaKernelY[KERNEL_SIZE * KERNEL_SIZE];

/* baseline, assign one thread to one pixel */
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
        for (int i = 0; i < KERNEL_SIZE; i ++) {
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

/* baseline + using constant memory for filter */
__global__ void convolveConstantFilter(unsigned char *image,
                                       int *convResult,
                                       int *convMax,
                                       int *convMin,
                                       int W,
                                       int H) {
    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int totalX = 0;
    int totalY = 0;

    if (imageX >= 0 && imageX < W && imageY >= 0 && imageY < H) {
        for (int i = 0; i < KERNEL_SIZE; i ++) {
            for (int j = 0; j < KERNEL_SIZE; j ++) {
                int currRow = imageY + i - 1;
                int currCol = imageX + j - 1;

                if (currRow >= 0 &&
                    currRow < H &&
                    currCol >= 0 &&
                    currCol < W) {
                    totalX += (int)(unsigned int)image[currRow * W + currCol] *
                              (int)cudaKernelX[i * KERNEL_SIZE + j];
                    totalY += (int)(unsigned int)image[currRow * W + currCol] *
                              (int)cudaKernelY[i * KERNEL_SIZE + j];
                }
            }
        }

        int currConvResult = sqrtf(totalX * totalX + totalY * totalY);
        convResult[imageY * W + imageX] = currConvResult;

        atomicMax(convMax, currConvResult);
        atomicMin(convMin, currConvResult);
    }
}

/* baseline + constant filter + shared memory for input image */
__global__ void convolveSharedMem(unsigned char *image,
                                  int *convResult,
                                  int *convMax,
                                  int *convMin,
                                  int W,
                                  int H) {

    __shared__ unsigned char blockImage[THREADS_PER_BLOCK];

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int totalX = 0;
    int totalY = 0;

    if (imageX < W && imageY < H) {
        blockImage[threadIdx.y * blockDim.x + threadIdx.x] = image[imageY * W + imageX];
    } else {
        blockImage[threadIdx.y * blockDim.x + threadIdx.x] = 0;
    }

    __syncthreads();

    if (imageX >= 0 && imageX < W && imageY >= 0 && imageY < H) {
        for (int i = 0; i < KERNEL_SIZE; i ++) {
            for (int j = 0; j < KERNEL_SIZE; j ++) {
                int blockRow = (int)threadIdx.y + i - 1;
                int blockCol = (int)threadIdx.x + j - 1;
                int imageRow = imageY + i - 1;
                int imageCol = imageX + j - 1;

                int pixel = 0;

                if (blockRow >= 0 &&
                    blockRow < (int)blockDim.y &&
                    blockCol >= 0 &&
                    blockCol < (int)blockDim.x) {
                    pixel = (int)(unsigned int)blockImage[blockRow * blockDim.x + blockCol];
                } else if (imageRow >= 0 &&
                           imageRow < H &&
                           imageCol >= 0 &&
                           imageCol < W) {

                    // the pixel resides in another block
                    pixel = (int)(unsigned int)image[imageRow * W + imageCol];
                }

                totalX += pixel * (int)cudaKernelX[i * KERNEL_SIZE + j];
                totalY += pixel * (int)cudaKernelY[i * KERNEL_SIZE + j];
            }
        }

        int currConvResult = sqrtf(totalX * totalX + totalY * totalY);
        convResult[imageY * W + imageX] = currConvResult;

        atomicMax(convMax, currConvResult);
        atomicMin(convMin, currConvResult);
    }
}

/* kernel to normalize the results from convolution */
__global__ void normalize(int *convResult,
                          unsigned char *outImage,
                          int convMax,
                          int convMin,
                          int W,
                          int H) {
    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    if (imageX >= 0 && imageX < W && imageY >= 0 && imageY < H) {
        outImage[imageY * W + imageX] =
            MAX_BRIGHTNESS * (convResult[imageY * W + imageX] - convMin) /
            (convMax - convMin);
    }
}

/* sobel baseline */
void sobelParBasic(unsigned char *inImage,
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

    cudaMemcpy(&initConvMax, convMax, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&initConvMin, convMin, sizeof(int), cudaMemcpyDeviceToHost);

    // normalize results
    normalize<<<gridDim, blockDim>>>(deviceConvResult,
                                     deviceImage,
                                     initConvMax,
                                     initConvMin,
                                     W,
                                     H);
    cudaDeviceSynchronize();

    cudaMemcpy(outImage, deviceImage, imageSize, cudaMemcpyDeviceToHost);
    cudaFree(deviceImage);
    cudaFree(deviceConvResult);
    cudaFree(convMax);
    cudaFree(convMin);
}

/* sobel improved */
void sobelPar(unsigned char *inImage,
              unsigned char *outImage,
              const char *kernelX,
              const char *kernelY,
              int W,
              int H) {

    unsigned char *deviceImage;
    int *deviceConvResult;
    int *convMax, *convMin;

    int initConvMax = INT_MIN;
    int initConvMin = INT_MAX;
    unsigned int imageSize = W * H * sizeof(unsigned char);
    unsigned int kernelSize = KERNEL_SIZE * KERNEL_SIZE * sizeof(char);
    unsigned int convResultSize = W * H * sizeof(int);

    cudaMalloc(&deviceImage, imageSize);
    cudaMalloc(&deviceConvResult, convResultSize);
    cudaMalloc(&convMax, sizeof(int));
    cudaMalloc(&convMin, sizeof(int));
    cudaMemcpy(deviceImage, inImage, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(convMax, &initConvMax, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(convMin, &initConvMin, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(cudaKernelX, kernelX, kernelSize);
    cudaMemcpyToSymbol(cudaKernelY, kernelY, kernelSize);

    dim3 blockDim(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);
    size_t gridWidth = (W + blockDim.x - 1) / blockDim.x;
    size_t gridHeight = (H + blockDim.y - 1) / blockDim.y;
    dim3 gridDim(gridWidth, gridHeight);

    // convolution
    convolveSharedMem<<<gridDim, blockDim>>>(deviceImage,
                                             deviceConvResult,
                                             convMax,
                                             convMin,
                                             W,
                                             H);
    cudaDeviceSynchronize();

    cudaMemcpy(&initConvMax, convMax, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&initConvMin, convMin, sizeof(int), cudaMemcpyDeviceToHost);

    // normalize results
    normalize<<<gridDim, blockDim>>>(deviceConvResult,
                                     deviceImage,
                                     initConvMax,
                                     initConvMin,
                                     W,
                                     H);
    cudaDeviceSynchronize();

    cudaMemcpy(outImage, deviceImage, imageSize, cudaMemcpyDeviceToHost);
    cudaFree(deviceImage);
    cudaFree(deviceConvResult);
    cudaFree(convMax);
    cudaFree(convMin);
}
