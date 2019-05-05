#include <convolution.hpp>
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


unsigned char *matToArray(cv::Mat &image) {

	int rows = image.rows;
	int cols = image.cols;

	unsigned char *ret = new unsigned char[rows * cols];

	for (int r = 0; r < rows; ++r) {
		for (int c = 0; c < cols; ++c) {
			ret[r * cols + c] = image.at<uchar>(r, c);
		}
	}

	return ret;
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

    cv::Mat image;
    image = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);   // Read the file

    unsigned char *imageArray = matToArray(image);

    sobel_seq(imageArray, sobelX.kernel, sobelY.kernel, image.cols, image.rows);
    // sobel(imageArray, sobelX.kernel, sobelY.kernel, image.cols, image.rows, sobelX.kx_offset, sobelX.ky_offset);
    // convolve(imageArray, sobelX.kernel, image.cols, image.rows, sobelX.kx_offset, sobelX.ky_offset);
    // convolve(imageArray, sobelY.kernel, image.cols, image.rows, sobelY.kx_offset, sobelY.ky_offset);
    cv::Mat filtered = arrayToMat(imageArray, image.rows, image.cols);

    std::cout << "rows: " << filtered.rows << ", cols: " << filtered.cols << std::endl;
    cv::imshow("bill", filtered);

    cv::waitKey(0);                                          // Wait for a keystroke in the window

	return 0;
}
