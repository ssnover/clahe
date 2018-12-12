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
struct TILE_COORDINATES
{
    unsigned int x;
    unsigned int y;
};

static void areaBasedGrayLevelMapping(IMAGE_HISTOGRAM const & histogram,
                                      LOOKUP_TABLE * outputTable);

static bool isCornerRegion(unsigned int x,
                           unsigned int y,
                           unsigned int tilesHorizontal,
                           unsigned int tileVertical,
                           cv::Mat const & input,
                           std::array<TILE_COORDINATES, 4> & outputTile);

static bool isBorderRegion(unsigned int x,
                           unsigned int y,
                           unsigned int tilesHorizontal,
                           unsigned int tilesVertical,
                           cv::Mat const & input,
                           std::array<TILE_COORDINATES, 4> & outputTile);

static bool getFourClosestTiles(unsigned int x,
                                unsigned int y,
                                unsigned int tilesHorizontal,
                                unsigned int tilesVertical,
                                cv::Mat const & input,
                                std::array<TILE_COORDINATES, 4> & outputTile);

static unsigned int getPixelCoordinateFromTileCoordinate(unsigned int tileCoordinate,
                                                         unsigned int pixelsInDimension,
                                                         unsigned int tilesInDimension);

[[nodiscard]] int clahe(cv::Mat const & input, cv::Mat & output, double clipLimit /* = 40.0 */) noexcept
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

            auto tileBounds = cv::Rect(tileWidth * colIdx, tileHeight * rowIdx,
                                       regionWidth, regionHeight);

            // Get the region of interest from the image
            cv::Mat regionOfInterest;
            getSubregionOfImage(input, tileBounds, regionOfInterest);

            // Get the histogram for the tile
            IMAGE_HISTOGRAM tileHistogram;
            generateGrayscaleHistogram(regionOfInterest, tileHistogram);

            // Clip the histogram and redistribute
            clipHistogram(tileHistogram, clipLimit);

            // Perform gray level mapping
            areaBasedGrayLevelMapping(
                tileHistogram, claheLookupTables->at<LOOKUP_TABLE *>(rowIdx, colIdx));
        }
    }

    // TODO: Interpolation between gray level mappings to form the output image
    // Now for each pixel, interpolate an intensity value from the gray level mappings of the closest tiles
    for (auto rowIdx = 0u; rowIdx < input.rows; ++rowIdx)
    {
        for (auto colIdx = 0u; colIdx < input.cols; ++colIdx)
        {
            // Find the four closest tile centers and the tile coordinates
            std::array<TILE_COORDINATES, 4> closestTiles{};
            if (isCornerRegion(colIdx, rowIdx, tilesHorizontal, tilesVertical, input, closestTiles))
            {
                // the closest tile is in index 0
                output.at<uint8_t>(rowIdx, colIdx) =
                    claheLookupTables
                        ->at<LOOKUP_TABLE *>(closestTiles[0].y, closestTiles[0].x)
                            ->operator[](input.at<uint8_t>(rowIdx, colIdx));
            }
            else if (isBorderRegion(colIdx, rowIdx, tilesHorizontal,
                                    tilesVertical, input, closestTiles))
            {
                // The two closest tiles are indices 0 and 1
                PIXEL pixel0 = {
                    // Get the tile center's xy-coordinates
                    getPixelCoordinateFromTileCoordinate(
                        closestTiles[0].x, input.cols, tilesHorizontal),
                    getPixelCoordinateFromTileCoordinate(
                        closestTiles[0].y, input.rows, tilesVertical),
                    // Plug the input pixel's intensity into the tile's lookup table
                    claheLookupTables
                        ->at<LOOKUP_TABLE *>(closestTiles[0].y, closestTiles[0].x)
                            ->operator[](input.at<uint8_t>(rowIdx, colIdx))};
                PIXEL pixel1 = {
                    // Get the tile center's xy-coordinates
                    getPixelCoordinateFromTileCoordinate(
                        closestTiles[1].x, input.cols, tilesHorizontal),
                    getPixelCoordinateFromTileCoordinate(
                        closestTiles[1].y, input.rows, tilesVertical),
                    // Plug the input pixel's intensity into the tile's lookup table
                    claheLookupTables
                        ->at<LOOKUP_TABLE *>(closestTiles[1].y, closestTiles[1].x)
                            ->operator[](input.at<uint8_t>(rowIdx, colIdx))};
                output.at<uint8_t>(rowIdx, colIdx) =
                    static_cast<uint8_t>(linearInterpolate(pixel0, pixel1).intensity);
            }
            else
            {
                // Grab the tile coordinates from all 4 indices
                getFourClosestTiles(colIdx, rowIdx, tilesHorizontal,
                                    tilesVertical, input, closestTiles);
                // Create a pixel for each tile center with an intensity from
                // mapping the current input pixel
                std::vector<PIXEL> tileCenters;
                for (auto & tile : closestTiles)
                {
                    tileCenters.push_back(
                        // Get the x-coordinate, y-coordinate of each tile center
                        {getPixelCoordinateFromTileCoordinate(tile.x, input.cols, tilesHorizontal),
                         getPixelCoordinateFromTileCoordinate(tile.y, input.rows, tilesVertical),
                         // Get the look-up table for this tile and plug this pixel's intensity in
                         claheLookupTables->at<LOOKUP_TABLE *>(tile.y, tile.x)
                             ->operator[](input.at<uint8_t>(rowIdx, colIdx))});
                }
                // Interpolate between the four pixels and assign it to the output image
                output.at<uint8_t>(rowIdx, colIdx) = static_cast<uint8_t>(
                    bilinearInterpolate(tileCenters, colIdx, rowIdx).intensity);
            }
            // Get the intensity value for each of these tile's mappings

            // Run bilinear interpolation to get the resulting new grayscale value

            // Assign it to the output image
        }
    }

    return -1;
}

static void areaBasedGrayLevelMapping(IMAGE_HISTOGRAM const & histogram, LOOKUP_TABLE * outputTable)
{
    unsigned int numberOfPixels(0);

    // Get the total number of pixels in the histogram
    for (auto i = 0u; i < 256; ++i)
    {
        numberOfPixels += histogram[i];
    }

    unsigned int numberOfPixelsSeen(0);
    for (auto i = 0u; i < outputTable->size(); ++i)
    {
        numberOfPixelsSeen += histogram[i];
        // How many of the pixels in this histogram have we seen?
        float ratioOfPixelsSeenToTotal = static_cast<float>(numberOfPixelsSeen) / numberOfPixels;
        // Readjust towards a more balanced image by moving pixels to where they "should" be
        outputTable->operator[](i) =
            static_cast<unsigned char>(ratioOfPixelsSeenToTotal * outputTable->size());
    }
}

} // namespace snover