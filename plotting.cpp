#include "plotting.hpp"
#include <opencv2/opencv.hpp>

int createHistogramPlot(ImageHistogram const & histogram,
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
                           static_cast<int>(height - histogram[i - 1] / verticalScaleFactor)), // point1
                 cv::Point(binWidth * i,
                           static_cast<int>(height - histogram[i] / verticalScaleFactor)), // point 2
                 cv::Scalar(255, 255, 255), // line color
                 2); // line thickness
    }

    return 0;
}

int createCDFPlot(ImageHistogram const & histogram,
                  unsigned int width,
                  unsigned int height,
                  cv::Mat & outputImage)
{
    unsigned int const numberOfBins(256);
    unsigned int const elementWidth(width / numberOfBins);
    float const verticalScaleFactor(static_cast<float>([&histogram]() {
                                        auto numberOfPixels = 0;
                                        for (auto i = 0u; i < 256; ++i)
                                        {
                                            numberOfPixels += histogram[i];
                                        }
                                        return numberOfPixels;
                                    }()) /
                                    height);

    outputImage = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    float cumulativeSum = histogram[0];
    for (auto i = 1u; i < numberOfBins; ++i)
    {
        cv::line(outputImage,
                 cv::Point(elementWidth * (i - 1),
                           static_cast<int>(height - cumulativeSum / verticalScaleFactor)),
                 cv::Point(elementWidth * i,
                           static_cast<int>(height - (cumulativeSum + histogram[i]) / verticalScaleFactor)),
                 cv::Scalar(255, 255, 255));
    }

    return 0;
}