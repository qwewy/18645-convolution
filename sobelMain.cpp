#include <time.h>
#include <sobel.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <stdlib.h>

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

inline void print_header(const char* s) {
    printf("===============================================================\n");
    printf("%s\n", s);
    printf("===============================================================\n");
}

inline void run_test(const char *header,
                     double seq_time,
                     int iterations,
                     void (*k)(unsigned char *inImage,
                               unsigned char *outImage,
                               const char *kernelX,
                               const char *kernelY,
                               int W,
                               int H),
                     unsigned char *inImage,
                     unsigned char *outImage,
                     const char *kernelX,
                     const char *kernelY,
                     int W,
                     int H
                     ){

    print_header(header);

    double total = 0.0f;
    for (int i = 0; i < iterations; i ++) {

        /* run and measure code performance */
        auto start = std::chrono::system_clock::now();
        k(inImage, outImage, kernelX, kernelY, W, H);
        auto end = std::chrono::system_clock::now();

        /* report runtime */
        std::chrono::duration<float> duration = end - start;
        printf("\tITER: %d, parallel baseline runtime: %.10f\n",
                i, duration.count());

        total += duration.count();
    }
    total /= iterations;
    printf("Average parallel baseline runtime: %.10f\n", total);
    printf("speedup: %.5fx\n", seq_time / total);
 
}

int main(int argc, char **argv) {

    double seq_time, parBasic_time, parConst_time, parShared_time;

    cv::Mat image;
    image = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);   // Read the file

    unsigned char inImage[image.rows * image.cols];
    unsigned char outImageSeq[image.rows * image.cols];
    unsigned char outImagePar[image.rows * image.cols];
    matToArray(image, inImage);

    int numIters = argc < 3 ? ITER : atoi(argv[2]);


    /**************************************************************************/
    /****************************** sequential ********************************/
    /**************************************************************************/

    print_header("sequential");

    seq_time = 0.f;
    for (int i = 0; i < numIters; i ++) {

        /* run and measure code performance */
        auto seq_start = std::chrono::system_clock::now();
        sobelSeq(inImage, outImageSeq, sobelX.kernel, 
                 sobelY.kernel, image.cols, image.rows);
        auto seq_end = std::chrono::system_clock::now();

        /* report runtime */
        std::chrono::duration<float> seq_duration = (seq_end - seq_start);
        printf("\tITER: %d, sequential runtime: %.10f\n", 
                i, seq_duration.count());

        seq_time += seq_duration.count();
    }
    seq_time /= numIters;
    printf("Average sequential time: %.10f\n", seq_time);


    /**************************************************************************/
    /**************************** parallel basic ******************************/
    /**************************************************************************/

    run_test("parallel basic", seq_time, numIters, sobelParBasic, inImage, 
             outImagePar, sobelX.kernel, sobelY.kernel, image.cols, image.rows);

    /**************************************************************************/
    /*********************** parallel constant kernel *************************/
    /**************************************************************************/

    run_test("parallel w/ const kernel", seq_time, numIters, 
             sobelParConstKernel, inImage, outImagePar, sobelX.kernel, 
             sobelY.kernel, image.cols, image.rows);

    /**************************************************************************/
    /************************** parallel shared mem ***************************/
    /**************************************************************************/

    run_test("parallel w/ shared mem", seq_time, numIters, 
             sobelParSharedMem, inImage, outImagePar, sobelX.kernel, 
             sobelY.kernel, image.cols, image.rows);

    /**************************************************************************/
    /*************************** correctness check ****************************/
    /**************************************************************************/

    print_header("correctness check");

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
