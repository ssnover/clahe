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
    std::string const filename("sqrt_boats.jpg");
    auto image = cv::imread(filename);

    cv::Mat processedImage;
    auto retVal = snover::clahe(image, processedImage);

    std::cout << "snover::clahe returned with " << retVal << std::endl;

    return 0;
}