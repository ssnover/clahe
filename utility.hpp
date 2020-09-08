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

enum GrayLevel : uint8_t
{
    LOW = 0,
    MIDDLE = 1,
    HIGH = 2,
};

struct Rectangle
{
    unsigned int x;
    unsigned int y;
    unsigned int width;
    unsigned int height;

    Rectangle(unsigned int _x, unsigned int _y, unsigned int _width, unsigned int _height)
      : x(_x), y(_y), width(_width), height(_height)
    {}
};

struct ImageHistogram
{
    std::vector<unsigned int> histogram;

    explicit ImageHistogram()
        : histogram(256, 0)
    {
        // Empty
    }

    friend GrayLevel classifyGrayLevel(ImageHistogram const & histogram);
    friend void clipHistogram(ImageHistogram & histogram, double clipLimit);

    inline unsigned int operator[](unsigned int index) const noexcept
    {
        return histogram[index];
    }

    unsigned int max() const
    {
        return *std::max_element(histogram.cbegin(), histogram.cend());
    }
};

struct Pixel
{
    unsigned int x;
    unsigned int y;
    unsigned int intensity;

    Pixel(unsigned int _x, unsigned int _y, unsigned int _intensity) : x(_x), y(_y), intensity(_intensity)
    {
        // Empty
    }

    bool operator<(Pixel const & rhs) const
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
int generateGrayscaleHistogram(cv::Mat const & image, ImageHistogram & outputHistogram);

/*
 * Generates the pixel intensity histogram for a subregion of a grayscale image.
 * 
 * image: An OpenCV matrix containing a grayscale image.
 * region: The subimage over which to create the histogram from.
 */
ImageHistogram generateGrayscaleHistogramForSubregion(cv::Mat const & image, Rectangle const & region);

/*
 * Classifies the image into one of three categories based on where the highest
 * number of gray scale intensities falls.
 *
 * Based on gray level definition of Youlian Zhu and Cheng Huang in "An Adaptive
 * Histogram Equalization Algorithm on the Image Gray Level Mapping".
 */
GrayLevel classifyGrayLevel(ImageHistogram const & histogram);

/*
 * Interpolates the value of a pixel based on it's linear distance in two
 * dimensions from four pixels.
 */
Pixel bilinearInterpolate(std::vector<Pixel> & pixels, float outX, float outY);

/*
 * Interpolates the value of a pixel based on it's linear distance in one
 * dimension from two pixels.
 */
Pixel linearInterpolate(Pixel pixel0, Pixel pixel1, float outX, float outY);

/*
 * Finds all bins of the histogram with a quantity over the clip limit and
 * removes the excess. The number of excess is added as equally as possible to
 * all bins in the histogram.
 */
void clipHistogram(ImageHistogram & histogram, double clipLimit);
