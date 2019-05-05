
#include <opencv2/opencv.hpp>




int main(int argc, char **argv) {

    cv::Mat image;
    image = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);   // Read the file


	return 0;
}