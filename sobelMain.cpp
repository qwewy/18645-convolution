#include <time.h>
#include <sobel.hpp>
#include <opencv2/opencv.hpp>

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

    clock_t start, end;

    cv::Mat image;
    image = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);   // Read the file

    unsigned char inImage[image.rows * image.cols];
    unsigned char outImageSeq[image.rows * image.cols];
    unsigned char outImagePar[image.rows * image.cols];
    matToArray(image, inImage);

    // TODO: add timing
    start = clock();
    sobelSeq(inImage, outImageSeq, sobelX.kernel, sobelY.kernel, image.cols, image.rows);
    end = clock();
    printf("sequential runtime: %.5f\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    start = clock();
    sobelPar(inImage, outImagePar, sobelX.kernel, sobelY.kernel, image.cols, image.rows);
    end = clock();
    printf("parallel runtime: %.5f\n", ((double)(end - start)) / CLOCKS_PER_SEC);

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
