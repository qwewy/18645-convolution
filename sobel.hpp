

#ifndef _SOBEL_H_
#define _SOBEL_H_

#define KERNEL_SIZE 3
#define MAX_BRIGHTNESS 255

void sobelParBasic(unsigned char *inImage,
                   unsigned char *outImage,
                   const char *kernelX,
                   const char *kernelY,
                   int W,
                   int H);

void sobelParConstKernel(unsigned char *inImage,
              unsigned char *outImage,
              const char *kernelX,
              const char *kernelY,
              int W,
              int H);

void sobelSeq(unsigned char *inImage,
              unsigned char *outImage,
              const char *kernelX,
              const char *kernelY,
              int W,
              int H);

void sobelParSharedMem(unsigned char *inImage,
              unsigned char *outImage,
              const char *kernelX,
              const char *kernelY,
              int W,
              int H);

#endif /* _SOBEL_H_ */
