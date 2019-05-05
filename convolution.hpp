

#ifndef _CONVOLUTION_H_
#define _CONVOLUTION_H_

void convolve(unsigned char *image, const char *kernel, int W, int H, int kx_offset, int ky_offset);


void sobel(unsigned char *image, const char *kernelX, const char *kernelY, int W, int H, int kx_offset, int ky_offset);


void sobel_seq(unsigned char *image, const char *kernelX, const char *kernelY, int W, int H);










#endif /* _CONVOLUTION_H_ */
