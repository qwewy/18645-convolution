#include <time.h>
#include <sobel.hpp>
#include <opencv2/opencv.hpp>

#define ITER 5

typedef struct {
	int kx_offset;
	int ky_offset;
	const char *kernel;
} filter_t;

const char sobelX_kernel[] = {-1, 0 ,1,
                              -2, 0, 2,
                              -1, 0, 1};

const char sobelY_kernel[] = {-1,  -2 , -1,
                               0,   0,   0,
                               1,   2,   1};

filter_t sobelX = {1, 1, sobelX_kernel};
filter_t sobelY = {1, 1, sobelY_kernel};


void matToArray(cv::Mat &image, unsigned char *inImage) {

	int rows = image.rows;
	int cols = image.cols;

	for (int r = 0; r < rows; ++r) {
		for (int c = 0; c < cols; ++c) {
			inImage[r * cols + c] = image.at<uchar>(r, c);
		}
	}
}

cv::Mat arrayToMat(unsigned char *image, int rows, int cols) {

	cv::Mat ret(rows, cols, CV_8UC1);

	for (int r = 0; r < rows; ++r) {
		for (int c = 0; c < cols; ++c) {
			ret.at<uchar>(r, c) = image[r * cols + c];
		}
	}

	return ret;
}


int main(int argc, char **argv) {

    clock_t seq_start, seq_end, par_start, par_end;
    double seq_time, par_time, curr_seq_time, curr_par_time;

    cv::Mat image;
    image = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);   // Read the file

    unsigned char inImage[image.rows * image.cols];
    unsigned char outImageSeq[image.rows * image.cols];
    unsigned char outImagePar[image.rows * image.cols];
    matToArray(image, inImage);

    for (int i = 0; i < ITER; i ++) {
        seq_start = clock();
        sobelSeq(inImage, outImageSeq, sobelX.kernel, sobelY.kernel, image.cols, image.rows);
        seq_end = clock();
        curr_seq_time = ((double)(seq_end - seq_start)) / CLOCKS_PER_SEC;
        printf("ITER: %d, sequential runtime: %.10f\n", i, curr_seq_time);

        seq_time += curr_seq_time;
    }
    seq_time /= ITER;
    printf("Average sequential time: %.10f\n", seq_time);

    par_time = 0.0f;
    for (int i = 0; i < ITER; i ++) {
        par_start = clock();
        sobelParBasic(inImage, outImagePar, sobelX.kernel, sobelY.kernel, image.cols, image.rows);
        par_end = clock();
        curr_par_time = ((double)(par_end - par_start)) / CLOCKS_PER_SEC;
        printf("ITER: %d, parallel baseline runtime: %.10f\n", i, curr_par_time);

        par_time += curr_par_time;
    }
    par_time /= ITER;
    printf("Average parallel baseline runtime: %.10f, speedup: %.5f\n", par_time, seq_time / par_time);

    par_time = 0.0f;
    for (int i = 0; i < ITER; i ++) {
        par_start = clock();
        sobelPar(inImage, outImagePar, sobelX.kernel, sobelY.kernel, image.cols, image.rows);
        par_end = clock();
        curr_par_time = ((double)(par_end - par_start)) / CLOCKS_PER_SEC;
        printf("ITER: %d, parallel improved runtime: %.10f\n", i, curr_par_time);

        par_time += curr_par_time;
    }
    par_time /= ITER;
    printf("Average parallel improved runtime: %.10f, speedup: %.5f\n", par_time, seq_time / par_time);

    std::cout << "Correctness check...\n";
    for (int row = 0; row < image.rows; row ++) {
      for (int col = 0; col < image.cols; col ++) {
        if (outImageSeq[row * image.cols + col] != outImagePar[row * image.cols + col]) {
          std::cout << "parallel results don't match with sequential results at row "
                    << row << ", col " << col << ", "
                    << "seq: " << (int)outImageSeq[row * image.cols + col]
                    << "par: " << (int)outImagePar[row * image.cols + col] << "\n";
          return -1;
        }
      }
    }
    std::cout << "Correctness check passed!\n";

    cv::Mat filtered = arrayToMat(outImageSeq, image.rows, image.cols);

    std::cout << "rows: " << filtered.rows << ", cols: " << filtered.cols << std::endl;
    cv::imshow("bill", filtered);

    cv::waitKey(0);                                          // Wait for a keystroke in the window

    return 0;
}
