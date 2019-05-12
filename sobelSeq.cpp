
#include <climits>
#include <math.h>
#include <sobel.hpp>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

int convolveSeq(unsigned char *image,
                const char *kernelX,
                const char *kernelY,
                int row,
                int col,
                int width,
                int height) {
    int totalX = 0;
    int totalY = 0;

    for (int i = 0; i < KERNEL_SIZE; i ++) {
        for (int j = 0; j < KERNEL_SIZE; j ++) {
            int currRow = row + i - 1;
            int currCol = col + j - 1;
            if (currRow >= 0 &&
                currRow < height &&
                currCol >= 0 &&
                currCol < width) {
                totalX += (int)image[currRow * width + currCol] *
                          (int)kernelX[i * KERNEL_SIZE + j];
                totalY += (int)image[currRow * width + currCol] *
                          (int)kernelY[i * KERNEL_SIZE + j];
            }
        }
    }

    return sqrt(totalX * totalX + totalY * totalY);
}

void sobelSeq(unsigned char *inImage,
              unsigned char *outImage,
              const char *kernelX,
              const char *kernelY,
              int W,
              int H) {
    int convMin = INT_MAX;
    int convMax = INT_MIN;
    int convResult[H * W];

    for (int row = 0; row < H; row ++) {
        for (int col = 0; col < W; col ++) {
            convResult[row * W + col] =
                convolveSeq(inImage, kernelX, kernelY, row, col, W, H);
            convMin = MIN(convResult[row * W + col], convMin);
            convMax = MAX(convResult[row * W + col], convMax);
        }
    }

    for (int row = 0; row < H; row ++) {
        for (int col = 0; col < W; col ++) {
            double resultNorm =
                (double)(convResult[row * W + col] - convMin) /
                (double)(convMax - convMin);
            outImage[row * W + col] = MAX_BRIGHTNESS * resultNorm;
        }
    }
}
