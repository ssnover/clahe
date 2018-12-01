/*
 * file: utility.hpp
 * purpose: Holds functions for doing various small manipulations, hiding the backend details.
 */

#pragma once

namespace cv
{
class Mat;
}

namespace snover
{

struct IMAGE_HISTOGRAM
{
    const std::string imageFilename;
    std::vector<unsigned int> * histogram;

    explicit IMAGE_HISTOGRAM(std::string _filename)
        : imageFilename(_filename), histogram(new std::vector<unsigned int>(256, 0))
    {
        // Empty
    }

    ~IMAGE_HISTOGRAM()
    {
        delete histogram;
    }

    inline unsigned int operator[](unsigned int index) const noexcept
    {
        return (*histogram)[index];
    }

    unsigned int max() const
    {
        return *std::max_element(histogram->begin(), histogram->end());
    }
};


/*
 * Generates the pixel intensity histogram for a grayscale image.
 *
 * image: An OpenCV matrix containing a grayscale image.
 * outputHistogram: Structure in which the output is to be populated.
 */
int generateGrayscaleHistogram(cv::Mat const & image, IMAGE_HISTOGRAM & outputHistogram);


int createHistogramPlot(IMAGE_HISTOGRAM const & histogram, unsigned int width, unsigned int height, cv::Mat & outputImage);


} // namespace snover