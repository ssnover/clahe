/*
 * file: opencv-clahe.cpp
 * purpose: Small application which takes an input image and applies OpenCV's
 *          histogram equalization implementation to the image.
 */

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"

int main(int argc, char ** argv)
{
    std::string inputImageFilename;

    if (argc >= 2)
    {
        inputImageFilename = argv[1];
    }
    else
    {
        return 2;
    }

    auto image = cv::imread(inputImageFilename);

    if (image.empty())
    {
        std::cout << "Unable to open the image: " << inputImageFilename << std::endl;
        return 1;
    }

    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    cv::Mat equalizedImage;
    auto clahe = cv::createCLAHE();
    clahe->apply(image, equalizedImage);

    std::string const windowNameOriginalImage("Original Image");
    std::string const windowNameNewImage("Histogram Equalized Image");

    cv::namedWindow(windowNameOriginalImage, cv::WINDOW_NORMAL);
    cv::namedWindow(windowNameNewImage, cv::WINDOW_NORMAL);

    cv::imshow(windowNameOriginalImage, image);
    cv::imshow(windowNameNewImage, equalizedImage);

    cv::waitKey(0);

    cv::destroyAllWindows();

    return 0;
}