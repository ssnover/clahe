/*
 * file: main.cpp
 * purpose: Implements a small executable which takes in an image filename from
 *          and applies a custom CLAHE algorithm to it before showing the new
 *          image with OpenCV's HighGUI.
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include "clahe.hpp"

int main()
{
    std::string const filename("/home/ssnover/develop/cv-631/clahe/sqrt_boats.jpg");
    auto image = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    cv::Mat processedImage;
    auto retVal = snover::clahe(image, processedImage);

    std::string const windowNameNewImage("Histogram Equalized Image");
    cv::namedWindow(windowNameNewImage, cv::WINDOW_NORMAL);
    cv::imshow(windowNameNewImage, processedImage);

    std::cout << "snover::clahe returned with " << retVal << std::endl;

    return 0;
}