
#include <climits>
#include <math.h>
#include <convolution.hpp>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

int convolve_seq(unsigned char *image,
                 const char *kernelX,
                 const char *kernelY,
                 int row,
                 int col,
                 int width,
                 int height) {
    int horizontal_total = 0;
    int vertical_total = 0;

    for (int i = 0; i < 3; i ++) { // TODO: Use macro
        for (int j = 0; j < 3; j ++) {
            int curr_row = row + i - 1;
            int curr_col = col + j - 1;
            if (curr_row >= 0 &&
                curr_row < height &&
                curr_col >= 0 &&
                curr_col < width) {
                horizontal_total += (int)image[curr_row * width + curr_col] *
                                    (int)kernelX[i * 3 + j];
                vertical_total += (int)image[curr_row * width + curr_col] *
                                  (int)kernelY[i * 3 + j];
            }
        }
    }

    return sqrt(horizontal_total * horizontal_total +
                vertical_total * vertical_total);
}

void sobel_seq(unsigned char *image,
               const char *kernelX,
               const char *kernelY,
               int W,
               int H) {
    int conv_min = INT_MAX;
    int conv_max = INT_MIN;
    int conv_result[H * W];

    for (int row = 0; row < H; row ++) {
        for (int col = 0; col < W; col ++) {
            conv_result[row * W + col] =
                convolve_seq(image, kernelX, kernelY, row, col, W, H);
            conv_min = MIN(conv_result[row * W + col], conv_min);
            conv_max = MAX(conv_result[row * W + col], conv_max);
        }
    }

    for (int row = 0; row < H; row ++) {
        for (int col = 0; col < W; col ++) {
            double result_norm =
                (double)(conv_result[row * W + col] - conv_min) /
                (double)(conv_max - conv_min);
            image[row * W + col] = 255 * result_norm;
        }
    }
}
