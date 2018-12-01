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

    // find the histogram of the original image
    snover::IMAGE_HISTOGRAM grayHistogram(inputImageFilename);
    snover::generateGrayscaleHistogram(image, grayHistogram);

    cv::Mat histogramImage;
    snover::createHistogramPlot(grayHistogram, 512, 512, histogramImage);

    // generate CLAHEd image
    cv::Mat equalizedImage;
    auto clahe = cv::createCLAHE();
    clahe->apply(image, equalizedImage);

    // find the histogram of the CLAHE image
    snover::IMAGE_HISTOGRAM claheHistogram("CLAHEd Image");
    snover::generateGrayscaleHistogram(equalizedImage, claheHistogram);
    cv::Mat claheHistImage;
    snover::createHistogramPlot(claheHistogram, 512, 512, claheHistImage);

    std::cout << "Entropies:" << std::endl;
    std::cout << "Original: " << snover::calculateEntropy(image) << std::endl;
    std::cout << "OpenCV CLAHE: " << snover::calculateEntropy(equalizedImage) << std::endl;

    std::string const windowNameOriginalImage("Original Image");
    std::string const windowNameNewImage("Histogram Equalized Image");
    std::string const windowNameOriginalHistogram("Histogram of Original Image");
    std::string const windowNameNewHistogram("Histogram of CLAHE Image");

    cv::namedWindow(windowNameOriginalImage, cv::WINDOW_NORMAL);
    cv::namedWindow(windowNameNewImage, cv::WINDOW_NORMAL);
    cv::namedWindow(windowNameOriginalHistogram, cv::WINDOW_NORMAL);
    cv::namedWindow(windowNameNewHistogram, cv::WINDOW_NORMAL);

    cv::imshow(windowNameOriginalImage, image);
    cv::imshow(windowNameNewImage, equalizedImage);
    cv::imshow(windowNameOriginalHistogram, histogramImage);
    cv::imshow(windowNameNewHistogram, claheHistImage);

    cv::waitKey(0);

    cv::destroyAllWindows();

    return 0;
}