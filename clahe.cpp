/*
 * file: clahe.cpp
 * purpose: Implementation of a generic adaptive histogram equalization algorithm.
 */

#include "clahe.hpp"
#include "opencv2/opencv.hpp"
#include "utility.hpp"

namespace snover
{

using LOOKUP_TABLE = std::array<uint8_t, 256>;

static void areaBasedGrayLevelMapping(IMAGE_HISTOGRAM const & histogram, LOOKUP_TABLE * outputTable);

[[nodiscard]] int clahe(cv::Mat const &input,
                        cv::Mat &output,
                        double clipLimit /* = 40.0 */) noexcept
{
    // Data on the tiles the image will be split into
    unsigned int const tilesHorizontal(8), tilesVertical(8);
    unsigned int const tileWidth(input.cols / tilesHorizontal);
    unsigned int const tileHeight(input.rows / tilesVertical);

    // Use CV_64F since the type's size will be the same as a pointer
    auto claheLookupTables = new cv::Mat(tilesVertical, tilesHorizontal, CV_64F);
    for (auto rowIdx = 0u; rowIdx < tilesVertical; ++rowIdx)
    {
        for (auto colIdx = 0u; colIdx < tilesHorizontal; ++colIdx)
        {
            claheLookupTables->at<LOOKUP_TABLE *>(rowIdx, colIdx) = new LOOKUP_TABLE;
        }
    }

    // Generate the look up table (mapping function) for each tile
    for (auto rowIdx = 0u; rowIdx < tilesVertical; ++rowIdx)
    {
        for (auto colIdx = 0u; colIdx < tilesHorizontal; ++colIdx)
        {
            // Determine the X-Y bounds of the tile
            unsigned int regionWidth = tileWidth;
            unsigned int regionHeight = tileHeight;
            // Grab the last few pixels if on the right edge of the image
            if (colIdx == tilesHorizontal - 1)
            {
                regionWidth += input.cols % tilesHorizontal;
            }
            // Grab the last few pixels if on the bottom edge of the image
            if (rowIdx == tilesVertical - 1)
            {
                regionHeight += input.rows % tilesVertical;
            }

            auto tileBounds = cv::Rect(tileWidth * colIdx, tileHeight * rowIdx, regionWidth, regionHeight);

            // Get the region of interest from the image
            cv::Mat regionOfInterest;
            getSubregionOfImage(input, tileBounds, regionOfInterest);

            // Get the histogram for the tile
            IMAGE_HISTOGRAM tileHistogram;
            generateGrayscaleHistogram(regionOfInterest, tileHistogram);

            // Clip the histogram and redistribute
            clipHistogram(tileHistogram, clipLimit);

            // Perform gray level mapping
            areaBasedGrayLevelMapping(tileHistogram, claheLookupTables->at<LOOKUP_TABLE *>(rowIdx, colIdx));
        }
    }

    // TODO: Interpolation between gray level mappings to form the output image

    return -1;
}

static void areaBasedGrayLevelMapping(IMAGE_HISTOGRAM const & histogram, LOOKUP_TABLE * outputTable)
{
    // TODO: Implement me!
}

} // namespace snover