#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

#define SOBEL_THRESHOLD 150

__inline__ __device__ unsigned char clamp(int x) {
	if (x > SOBEL_THRESHOLD) {
		x = 0;
	} else if (x < 0) {
		x = 0;
	}
	return x;
}

/* assign thread per pixel, and compute individually */
__global__ void convolve_basic(unsigned char *image, char *kernel, int W, int H, int kx_offset, int ky_offset) {

	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int kw = kx_offset*2 + 1;
    // int kh = ky_offset*2 + 1;
    int total = 0;

    if (imageX < W && imageY < H) {
    	for (int ky = -ky_offset; ky <= ky_offset; ++ky) {
    		for (int kx = 0; kx <= kx_offset; ++kx) {
    			int ix = imageX + kx;
    			int iy = imageY + ky;
    			if (ix < W && iy < H && ix >= 0 && iy >= 0) {
    				total += (int)(unsigned int)image[iy * W + ix] * (int)kernel[ky * kw + kx];
    			}
    		}
    	}
    }

    image[imageY*W + imageX] = clamp(total);
}

void convolve(unsigned char *image, const char *kernel, int W, int H, int kx_offset, int ky_offset) {

    unsigned char *deviceImage;
    char *deviceKernel;

    int kernelNumElems = (2 * kx_offset + 1) * (2 * ky_offset + 1);
    int imageSize = W * H * sizeof(unsigned char);
    int kernelSize = kernelNumElems * sizeof(char);

    cudaMalloc(&deviceImage, imageSize);
    cudaMalloc(&deviceKernel, kernelSize);
    cudaMemcpy(deviceImage, image, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceKernel, kernel, kernelSize, cudaMemcpyHostToDevice);

    dim3 blockDim(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);
    size_t gridWidth = (W + blockDim.x - 1) / blockDim.x;
    size_t gridHeight = (H + blockDim.y - 1) / blockDim.y;
    dim3 gridDim(gridWidth, gridHeight);

    convolve_basic<<<gridDim, blockDim>>>(deviceImage, deviceKernel, W, H, kx_offset, ky_offset);
    cudaDeviceSynchronize();

    cudaMemcpy(image, deviceImage, imageSize, cudaMemcpyDeviceToHost);


}


/* assign thread per pixel, and compute individually */
__global__ void sobel_basic(unsigned char *image, char *kernelX, char *kernelY, int W, int H, int kx_offset, int ky_offset) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int kw = kx_offset*2 + 1;
    // int kh = ky_offset*2 + 1;
    int totalX = 0;
    int totalY = 0;

    if (imageX < W && imageY < H) {
        for (int ky = -ky_offset; ky <= ky_offset; ++ky) {
            for (int kx = 0; kx <= kx_offset; ++kx) {
                int ix = imageX + kx;
                int iy = imageY + ky;
                if (ix < W && iy < H && ix >= 0 && iy >= 0) {
                    totalX += (int)(unsigned int)image[iy * W + ix] * (int)kernelX[ky * kw + kx];
                    totalY += (int)(unsigned int)image[iy * W + ix] * (int)kernelY[ky * kw + kx];
                }
            }
        }
    }



    image[imageY*W + imageX] = clamp((int)sqrtf(powf((float)totalX, 2) + powf((float)totalY, 2)));
}

void sobel(unsigned char *image, const char *kernelX, const char *kernelY, int W, int H, int kx_offset, int ky_offset) {

    unsigned char *deviceImage;
    char *deviceKernelX, *deviceKernelY;

    int kernelNumElems = (2 * kx_offset + 1) * (2 * ky_offset + 1);
    int imageSize = W * H * sizeof(unsigned char);
    int kernelSize = kernelNumElems * sizeof(char);

    cudaMalloc(&deviceImage, imageSize);
    cudaMalloc(&deviceKernelX, kernelSize);
    cudaMalloc(&deviceKernelY, kernelSize);
    cudaMemcpy(deviceImage, image, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceKernelX, kernelX, kernelSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceKernelY, kernelY, kernelSize, cudaMemcpyHostToDevice);

    dim3 blockDim(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);
    size_t gridWidth = (W + blockDim.x - 1) / blockDim.x;
    size_t gridHeight = (H + blockDim.y - 1) / blockDim.y;
    dim3 gridDim(gridWidth, gridHeight);

    sobel_basic<<<gridDim, blockDim>>>(deviceImage, deviceKernelX, deviceKernelY, W, H, kx_offset, ky_offset);
    cudaDeviceSynchronize();

    cudaMemcpy(image, deviceImage, imageSize, cudaMemcpyDeviceToHost);


}





