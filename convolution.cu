#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

__inline__ __device__ char clamp(int x) {
	if (x > 255) {
		x = 255;
	} else if (x < 0) {
		x = 0;
	}
	return x;
}

/* assign thread per pixel, and compute individually */
__global__ void convolve_basic(char *image, char *kernel, int W, int H, int kx_offset, int ky_offset) {

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
    				total += image[iy * W + ix] * kernel[ky * kw + kx];
    			}
    		}
    	}
    }

    image[imageY*W + imageX] = clamp(total);
}