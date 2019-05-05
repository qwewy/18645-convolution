

#ifndef _CONVOLUTION_H_
#define _CONVOLUTION_H_

void convolve(char *image, const char *kernel, int W, int H, int kx_offset, int ky_offset);


void sobel(char *image, const char *kernelX, const char *kernelY, int W, int H, int kx_offset, int ky_offset);













#endif /* _CONVOLUTION_H_ */