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

int main(int argc, char **argv) {

    double seq_time, parBasic_time, par_time;

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

    print_header("parallel basic");

    parBasic_time = 0.0f;
    for (int i = 0; i < numIters; i ++) {

        /* run and measure code performance */
        auto parBasic_start = std::chrono::system_clock::now();
        sobelParBasic(inImage, outImagePar, sobelX.kernel, 
                      sobelY.kernel, image.cols, image.rows);
        auto parBasic_end = std::chrono::system_clock::now();

        /* report runtime */
        std::chrono::duration<float> parBasic_duration;
        parBasic_duration = parBasic_end - parBasic_start;
        printf("\tITER: %d, parallel baseline runtime: %.10f\n",
                i, parBasic_duration.count());

        parBasic_time += parBasic_duration.count();
    }
    parBasic_time /= numIters;
    printf("Average parallel baseline runtime: %.10f\n", parBasic_time);
    printf("speedup: %.5fx\n", seq_time / parBasic_time);

    /**************************************************************************/
    /******************************* parallel *********************************/
    /**************************************************************************/

    print_header("prallel");

    par_time = 0.0f;
    for (int i = 0; i < numIters; i ++) {

        /* run and measure code performance */
        auto par_start = std::chrono::system_clock::now();
        sobelPar(inImage, outImagePar, sobelX.kernel, 
                 sobelY.kernel, image.cols, image.rows);
        auto par_end = std::chrono::system_clock::now();

        /* report runtime */
        std::chrono::duration<float> par_duration = (par_end - par_start);
        printf("\tITER: %d, parallel improved runtime: %.10f\n", 
                i, par_duration.count());
        par_time += par_duration.count();
    }
    par_time /= numIters;

    printf("Average parallel improved runtime: %.10f\n", par_time);
    printf("speedup: %.5fx\n", seq_time / par_time);

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
