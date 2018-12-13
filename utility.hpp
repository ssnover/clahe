/*
 * file: utility.hpp
 * purpose: Holds functions for doing various small manipulations, hiding the backend details.
 */

#pragma once

#include <cstdint>
#include <opencv2/core/types.hpp>
#include <vector>

namespace cv
{
class Mat;
} // namespace cv

namespace snover
{
enum GRAY_LEVEL : uint8_t
{
    LOW = 0,
    MIDDLE = 1,
    HIGH = 2,
};

struct IMAGE_HISTOGRAM
{
    std::vector<unsigned int> * histogram;

    explicit IMAGE_HISTOGRAM()
        : histogram(new std::vector<unsigned int>(256, 0))
    {
        // Empty
    }

    ~IMAGE_HISTOGRAM()
    {
        delete histogram;
    }

    friend GRAY_LEVEL classifyGrayLevel(IMAGE_HISTOGRAM const & histogram);
    friend void clipHistogram(IMAGE_HISTOGRAM & histogram, double clipLimit);

    inline unsigned int operator[](unsigned int index) const noexcept
    {
        return (*histogram)[index];
    }

    unsigned int max() const
    {
        return *std::max_element(histogram->begin(), histogram->end());
    }
};

struct PIXEL
{
    unsigned int x;
    unsigned int y;
    unsigned int intensity;

    PIXEL(unsigned int _x, unsigned int _y, unsigned int _intensity) : x(_x), y(_y), intensity(_intensity)
    {
        // Empty
    }

    bool operator<(PIXEL const & rhs) const
    {
        if (this->x < rhs.x)
        {
            return true;
        }
        else
        {
            return this->y < rhs.y;
        }
    }
};

/*
 * Generates the pixel intensity histogram for a grayscale image.
 *
 * image: An OpenCV matrix containing a grayscale image.
 * outputHistogram: Structure in which the output is to be populated.
 */
int generateGrayscaleHistogram(cv::Mat const & image, IMAGE_HISTOGRAM & outputHistogram);

/*
 * Takes in a histogram and parameters for the size of the output plot image
 * and creates the plot image.
 */
int createHistogramPlot(IMAGE_HISTOGRAM const & histogram,
                        unsigned int width,
                        unsigned int height,
                        cv::Mat & outputImage);

/*
 * Takes in a histogram and parameters for the size of the output plot image
 * and creates a plot image of a CDF for the histogram.
 */
int createCDFPlot(IMAGE_HISTOGRAM const & histogram,
                  unsigned int width,
                  unsigned int height,
                  cv::Mat & outputImage);

/*
 * Calculates the entropy measurement of a grayscale image.
 */
float calculateEntropy(cv::Mat const & image);

/*
 * Gets a rectangular subregion of the image.
 */
int getSubregionOfImage(cv::Mat const & input, cv::Rect const & region, cv::Mat & output);

/*
 * Classifies the image into one of three categories based on where the highest
 * number of gray scale intensities falls.
 *
 * Based on gray level definition of Youlian Zhu and Cheng Huang in "An Adaptive
 * Histogram Equalization Algorithm on the Image Gray Level Mapping".
 */
GRAY_LEVEL classifyGrayLevel(IMAGE_HISTOGRAM const & histogram);

/*
 * Interpolates the value of a pixel based on it's linear distance in two
 * dimensions from four pixels.
 */
PIXEL bilinearInterpolate(std::vector<PIXEL> & pixels, float outX, float outY);

/*
 * Interpolates the value of a pixel based on it's linear distance in one
 * dimension from two pixels.
 *
 * TODO: Linear interpolation is completely incorrect right now...
 */
PIXEL linearInterpolate(PIXEL pixel0, PIXEL pixel1, float outX, float outY);

/*
 * Finds all bins of the histogram with a quantity over the clip limit and
 * removes the excess. The number of excess is added as equally as possible to
 * all bins in the histogram.
 */
void clipHistogram(IMAGE_HISTOGRAM & histogram, double clipLimit);

} // namespace snover