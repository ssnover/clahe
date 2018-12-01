/*
 * file: opencv-clahe.cpp
 * purpose: Small application which takes an input image and applies OpenCV's
 *          histogram equalization implementation to the image.
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "utility.hpp"
#include <iostream>

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

    auto image = cv::imread(inputImageFilename, cv::IMREAD_GRAYSCALE);

    if (image.empty())
    {
        std::cout << "Unable to open the image: " << inputImageFilename << std::endl;
        return 1;
    }

    //cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    // find the histogram of the original image
    int numberOfBins(256);
    snover::IMAGE_HISTOGRAM grayHistogram(inputImageFilename);
    snover::generateGrayscaleHistogram(image, grayHistogram);

    cv::Mat histogramImage;
    snover::createHistogramPlot(grayHistogram, 512, 512, histogramImage);

    cv::Mat equalizedImage;
    auto clahe = cv::createCLAHE();
    clahe->apply(image, equalizedImage);

    std::string const windowNameOriginalImage("Original Image");
    std::string const windowNameNewImage("Histogram Equalized Image");

    cv::namedWindow(windowNameOriginalImage, cv::WINDOW_NORMAL);
    cv::namedWindow(windowNameNewImage, cv::WINDOW_NORMAL);
    cv::namedWindow("Histogram of Original Image", cv::WINDOW_NORMAL);

    cv::imshow(windowNameOriginalImage, image);
    cv::imshow(windowNameNewImage, equalizedImage);
    cv::imshow("Histogram of Original Image", histogramImage);

    cv::waitKey(0);

    cv::destroyAllWindows();

    return 0;
}