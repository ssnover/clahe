/*
 * file: clahe.hpp
 * purpose: Declaration of a free function which takes images read in through
 *          OpenCV and performs a contrast-limited adaptive histogram
 *          equalization operation.
 */

#pragma once

#include <functional>
#include "utility.hpp"

/*
 * Forward Declarations
 */
namespace cv
{
class Mat;
}

/*
 * clahe.hpp Declarations
 */
namespace snover
{

using LookupTable = std::array<uint8_t, 256>;

using GrayLevelMappingFunction = std::function<void(ImageHistogram const & histogram, LookupTable * outputTable)>;

/*
 * Takes a grayscale image and runs a CLAHE algorithm on it.
 *
 * input- The matrix holding the input image.
 * output- The matrix for the output image to be stored in.
 * clipLimit- The limit for a single bin of the histogram.
 *
 * Returns an integer indicating whether the process was successful; 0
 * indicates success and -1 indicates a failure.
 */
[[nodiscard]] int clahe(cv::Mat const & input,
                     cv::Mat & output,
                     double clipLimit = 40.0) noexcept;

[[nodiscard]] int clahe(cv::Mat const & input,
                        cv::Mat & output,
                        GrayLevelMappingFunction mapping,
                        double clipLimit = 40.0) noexcept;

} // namespace snover