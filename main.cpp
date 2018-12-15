/*
 * file: main.cpp
 * purpose: Implements a small executable which takes in an image filename from
 *          and applies a custom CLAHE algorithm to it before showing the new
 *          image with OpenCV's HighGUI.
 */

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "clahe.hpp"
#include "utility.hpp"

static void unityMapping(snover::IMAGE_HISTOGRAM const & histogram, snover::LOOKUP_TABLE * outputTable)
{
    for (auto i = 0u; i < outputTable->size(); ++i)
    {
        outputTable->operator[](i) = static_cast<uint8_t >(i);
    }
}

int main(int argc, char ** argv)
{
    if (argc < 2)
    {
        std::cerr << "Must provide an input image." << std::endl;
        return 1;
    }

    auto image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    cv::Mat processedImage;
    int retVal(-1);

    auto start = std::chrono::high_resolution_clock::now();
    if (argc == 3)
    {
        retVal = snover::clahe(image, processedImage, atoi(argv[2]));
    }
    else
    {
        // In order to specify your own mapping function:
        // retVal = snover::clahe(image, processedImage, unityMapping);
        retVal = snover::clahe(image, processedImage);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Duration (us): " << duration.count() << std::endl;

    std::string const windowNameNewImage("Histogram Equalized Image");
    cv::namedWindow(windowNameNewImage, cv::WINDOW_NORMAL);
    cv::imshow(windowNameNewImage, processedImage);

    // Get the histogram of the original image
    snover::IMAGE_HISTOGRAM originalHistogram;
    snover::generateGrayscaleHistogram(image, originalHistogram);
    cv::Mat originalHistogramImage;
    snover::createHistogramPlot(originalHistogram, 512, 512, originalHistogramImage);
    // Display it
    std::string const windowOriginalHistogram("Original Histogram");
    cv::namedWindow(windowOriginalHistogram, cv::WINDOW_NORMAL);
    cv::imshow(windowOriginalHistogram, originalHistogramImage);

    // Get the histogram of the new image
    snover::IMAGE_HISTOGRAM claheHistogram;
    snover::generateGrayscaleHistogram(processedImage, claheHistogram);
    cv::Mat claheHistImage;
    snover::createHistogramPlot(claheHistogram, 512, 512, claheHistImage);
    // Display it
    std::string const windowNewHistogram("New Histogram");
    cv::namedWindow(windowNewHistogram, cv::WINDOW_NORMAL);
    cv::imshow(windowNewHistogram, claheHistImage);

    cv::waitKey(0);

    cv::destroyAllWindows();

    cv::imwrite("output-clahe.jpg", processedImage);

    std::cout << "snover::clahe returned with " << retVal << std::endl;

    return 0;
}