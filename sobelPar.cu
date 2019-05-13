#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <math.h>
#include <sobel.hpp>
#include <stdio.h>
#include <assert.h>

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32
#define THREADS_PER_BLOCK 1024

#define SHARED_ROWS 34
#define SHARED_COLS 34
#define SHARED_TOTAL 1156

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

    __shared__ unsigned char blockImage[SHARED_TOTAL];

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;
    int imageIndex = imageY * W + imageX;

    int totalX = 0;
    int totalY = 0;

    int blockIndexX = threadIdx.x + 1;
    int blockIndexY = threadIdx.y + 1;
    int blockIndex = blockIndexY * SHARED_COLS + blockIndexX;

    if (imageX < W && imageY < H) {
        blockImage[blockIndex] = image[imageY * W + imageX];

        // topleft pixel, find the pixels to its left, top and upperleft
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            // pixel to the upperleft
            blockImage[0] = (imageX == 0 || imageY == 0) ? 0 :
                            image[imageIndex - W - 1];

            // pixel to the left
            blockImage[blockIndex - 1] =
                (imageX == 0) ? 0 : image[imageIndex - 1];

            // pixel to the top
            blockImage[blockIndexX] =
                (imageY == 0) ? 0 : image[imageIndex - W];
        }

        // bottomright pixel
        else if ((threadIdx.x == blockDim.x - 1 || imageX == W - 1) &&
                 (threadIdx.y == blockDim.y - 1 || imageY == H - 1)) {
            // pixel to the bottomright
            blockImage[blockIndex + SHARED_COLS + 1] =
                (imageY == H - 1 || imageX == W - 1) ? 0 : image[imageIndex + W + 1];

            // pixel to the right
            blockImage[blockIndex + 1] =
                (imageX == W - 1) ? 0 : image[imageIndex + 1];

            // pixel to the bottom
            blockImage[blockIndex + SHARED_COLS] =
                (imageY == H - 1) ? 0 : image[imageIndex + W];
        }

        // bottomleft pixel
        else if (threadIdx.x == 0 &&
                 (threadIdx.y == blockDim.y - 1 || imageY == H - 1)) {
            // pixel to the bottomleft
            blockImage[blockIndex + SHARED_COLS - 1] =
                (imageY == H - 1 || imageX == 0) ? 0 : image[imageIndex + W - 1];

            // pixel to the bottom
            blockImage[blockIndex + SHARED_COLS] =
                (imageY == H - 1) ? 0 : image[imageIndex + W];

            // pixel to the left
            blockImage[blockIndex - 1] =
                (imageX == 0) ? 0 : image[imageIndex - 1];
        }

        // topright pixel
        else if ((threadIdx.x == blockDim.x - 1 || imageX == W - 1) &&
                 threadIdx.y == 0) {
            // pixel to the upperright
            blockImage[blockIndexX + 1] =
                (imageY == 0 || imageX == W - 1) ? 0 : image[imageIndex - W + 1];

            // pixel to the top
            blockImage[blockIndexX] =
                (imageY == 0) ? 0 : image[imageIndex - W];

            // pixel to the right
            blockImage[blockIndex + 1] =
                (imageX == W - 1) ? 0 : image[imageIndex + 1];
        }

        // if the pixel is in the leftmost column, find the pixel value to
        // its left in the image or 0
        else if (threadIdx.x == 0) {
            blockImage[blockIndex - 1] =
                (imageX == 0) ? 0 : image[imageIndex - 1];
        }

        // if the pixel is in the top row, find the pixel value to its top
        // in the image or 0
        else if (threadIdx.y == 0) {
            blockImage[blockIndexX] =
                (imageY == 0) ? 0 : image[imageIndex - W];
        }

        // if the pixel is in the rightmost column, find the pixel value
        // to its right in the image or 0
        else if (threadIdx.x == blockDim.x - 1 || imageX == W - 1) {
            blockImage[blockIndex + 1] =
                (imageX == W - 1) ? 0 : image[imageIndex + 1];
        }

        // if the pixel is in the bottom row, find the pixel value
        // to its bottom or 0
        else if (threadIdx.y == blockDim.y - 1 || imageY == H - 1) {
            blockImage[blockIndex + SHARED_COLS] =
                (imageY == H - 1) ? 0 : image[imageIndex + W];
        }

    } else {
        blockImage[blockIndex] = 0;
    }

    __syncthreads();

    if (imageX >= 0 && imageX < W && imageY >= 0 && imageY < H) {
        for (int i = 0; i < KERNEL_SIZE; i ++) {
            for (int j = 0; j < KERNEL_SIZE; j ++) {
                int blockRow = blockIndexY + i - 1;
                int blockCol = blockIndexX + j - 1;

                int pixel = (int)(unsigned int)blockImage[blockRow * SHARED_COLS + blockCol];
                totalX += pixel * (int)cudaKernelX[i * KERNEL_SIZE + j];
                totalY += pixel * (int)cudaKernelY[i * KERNEL_SIZE + j];
            }
        }

        int currConvResult = sqrtf(totalX * totalX + totalY * totalY);
        convResult[imageIndex] = currConvResult;

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
void sobelParConstKernel(unsigned char *inImage,
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
    convolveConstantFilter<<<gridDim, blockDim>>>(deviceImage,
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


/* sobel improved */
void sobelParSharedMem(unsigned char *inImage,
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
