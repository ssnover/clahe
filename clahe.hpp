/*
 * file: clahe.hpp
 * purpose: Declaration of a free function which takes images read in through
 *          OpenCV and performs a contrast-limited adaptive histogram
 *          equalization operation.
 */

#pragma once

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

/*
 * Type for representing the dimensions of the tile grid used in CLAHE.
 */
struct TILE_GRID_DIMENSIONS
{
    // The horizontal dimension in number of pixel columns.
    unsigned int x;
    // The vertical dimension in number of pixel rows.
    unsigned int y;
};

/*
 * Takes a grayscale image and runs a CLAHE algorithm on it.
 *
 * input- The matrix holding the input image.
 * output- The matrix for the output image to be stored in.
 * clipLimit- The limit for a single bin of the histogram.
 * tileDimensions- The row and column dimensions used for tiling the local
 *                 regions.
 *
 * Returns an integer indicating whether the process was successful; 0
 * indicates success and -1 indicates a failure.
 */
[[nodiscard]] constexpr int clahe(cv::Mat const & input,
                     cv::Mat & output,
                     double clipLimit = 40.0,
                     TILE_GRID_DIMENSIONS const & tileDimensions = {8, 8}) noexcept
{
    return -1;
}

} // namespace snover