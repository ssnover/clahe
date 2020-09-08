#pragma once

#include "utility.hpp"

namespace cv
{
class Mat;
} // namespace cv

/*
 * Takes in a histogram and parameters for the size of the output plot image
 * and creates the plot image.
 */
int createHistogramPlot(ImageHistogram const & histogram,
                        unsigned int width,
                        unsigned int height,
                        cv::Mat & outputImage);

/*
 * Takes in a histogram and parameters for the size of the output plot image
 * and creates a plot image of a CDF for the histogram.
 */
int createCDFPlot(ImageHistogram const & histogram,
                  unsigned int width,
                  unsigned int height,
                  cv::Mat & outputImage);