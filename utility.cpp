/*
 * file: utility.cpp
 */

#include <opencv2/opencv.hpp>
#include "utility.hpp"

namespace snover
{

int generateGrayscaleHistogram(cv::Mat const & image, IMAGE_HISTOGRAM & outputHistogram)
{
    if (outputHistogram.histogram->size() != 256)
    {
        return -1;
    }

    for (auto rowIdx = 0u; rowIdx < image.rows; ++rowIdx)
    {
        for (auto colIdx = 0u; colIdx < image.cols; ++colIdx)
        {
            (*(outputHistogram.histogram))[image.at<uint8_t>(rowIdx, colIdx)]++;
        }
    }

    return 0;
}

int createHistogramPlot(IMAGE_HISTOGRAM const & histogram,
                        unsigned int width,
                        unsigned int height,
                        cv::Mat & outputImage)
{
    unsigned int const numberOfBins(256);
    unsigned int const binWidth(width / numberOfBins);
    float const verticalScaleFactor(static_cast<float>(histogram.max()) / height);

    outputImage = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    for (auto i = 1u; i < numberOfBins; ++i)
    {
        cv::line(outputImage,
                 cv::Point(binWidth * (i - 1),
                           static_cast<int>(height - histogram[i-1] / verticalScaleFactor)), // point1
                 cv::Point(binWidth * i,
                           static_cast<int>(height - histogram[i] / verticalScaleFactor)), // point 2
                 cv::Scalar(255, 255, 255)); // line color
    }

    return 0;
}

float calculateEntropy(cv::Mat const & image)
{
    IMAGE_HISTOGRAM temp("");
    generateGrayscaleHistogram(image, temp);

    auto totalPixels(image.rows * image.cols);
    float entropy = 0.0;

    for (auto i = 0u; i < 256; ++i)
    {
        float proportion = static_cast<float>(temp[i]) / totalPixels;
        entropy += -1 * proportion * log2(proportion);
    }

    return entropy;
}

} // namespace snover