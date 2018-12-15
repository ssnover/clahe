/*
 * file: opencv-clahe.cpp
 * purpose: Small application which takes an input image and applies OpenCV's
 *          histogram equalization implementation to the image.
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "utility.hpp"
#include <iostream>
#include <chrono>

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
    snover::IMAGE_HISTOGRAM grayHistogram;
    snover::generateGrayscaleHistogram(image, grayHistogram);

    cv::Mat histogramImage;
    snover::createHistogramPlot(grayHistogram, 512, 512, histogramImage);

    // do normal OpenCV histogram equalization
    cv::Mat normalEqualization;
    cv::equalizeHist(image, normalEqualization);

    // generate CLAHEd image
    cv::Mat claheImage;
    auto start = std::chrono::high_resolution_clock::now();
    auto clahe = cv::createCLAHE();
    clahe->apply(image, claheImage);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Duration (us): " << duration.count() << std::endl;

    // find the histogram of the CLAHE image
    snover::IMAGE_HISTOGRAM claheHistogram;
    snover::generateGrayscaleHistogram(claheImage, claheHistogram);
    cv::Mat claheHistImage;
    snover::createHistogramPlot(claheHistogram, 512, 512, claheHistImage);

    std::cout << "Entropies:" << std::endl;
    std::cout << "Original: " << snover::calculateEntropy(image) << std::endl;
    std::cout << "OpenCV CLAHE: " << snover::calculateEntropy(claheImage) << std::endl;
    std::cout << "OpenCV Normal histeq: " << snover::calculateEntropy(normalEqualization) << std::endl;

    std::string const windowNameOriginalImage("Original Image");
    std::string const windowNameNewImage("Histogram Equalized Image");
    std::string const windowNameOriginalHistogram("Histogram of Original Image");
    std::string const windowNameNewHistogram("Histogram of CLAHE Image");

    cv::namedWindow(windowNameOriginalImage, cv::WINDOW_NORMAL);
    cv::namedWindow(windowNameNewImage, cv::WINDOW_NORMAL);
    cv::namedWindow(windowNameOriginalHistogram, cv::WINDOW_NORMAL);
    cv::namedWindow(windowNameNewHistogram, cv::WINDOW_NORMAL);

    cv::imshow(windowNameOriginalImage, image);
    cv::imshow(windowNameNewImage, claheImage);
    cv::imshow(windowNameOriginalHistogram, histogramImage);
    cv::imshow(windowNameNewHistogram, claheHistImage);

    cv::waitKey(0);

    cv::destroyAllWindows();

    return 0;
}