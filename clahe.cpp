/*
 * file: clahe.cpp
 * purpose: Implementation of a generic adaptive histogram equalization algorithm.
 */

#include <array>
#include "opencv2/opencv.hpp"
#include "clahe.hpp"

namespace snover
{

struct TileCoordinates
{
    unsigned int x;
    unsigned int y;
};

static void areaBasedGrayLevelMapping(ImageHistogram const & histogram,
                                      LookupTable * outputTable);

static bool isCornerRegion(unsigned int x,
                           unsigned int y,
                           unsigned int tilesHorizontal,
                           unsigned int tilesVertical,
                           cv::Mat const & input,
                           std::array<TileCoordinates, 4> & outputTile);

static bool isBorderRegion(unsigned int x,
                           unsigned int y,
                           unsigned int tilesHorizontal,
                           unsigned int tilesVertical,
                           cv::Mat const & input,
                           std::array<TileCoordinates, 4> & outputTile);

static void getFourClosestTiles(unsigned int x,
                                unsigned int y,
                                unsigned int tilesHorizontal,
                                unsigned int tilesVertical,
                                cv::Mat const & input,
                                std::array<TileCoordinates, 4> & outputTile);

static unsigned int getPixelCoordinateFromTileCoordinate(unsigned int tileCoordinate,
                                                         unsigned int pixelsPerTile);

static unsigned int getLowerTileCoordinate(float pixelDimension, float tileDimension);

[[nodiscard]] int clahe(cv::Mat const & input, cv::Mat & output, double clipLimit /* = 40.0 */) noexcept
{
    return clahe(input, output, areaBasedGrayLevelMapping, clipLimit);
}

[[nodiscard]] int clahe(cv::Mat const & input, cv::Mat & output, GrayLevelMappingFunction mapping, double clipLimit /* = 40.0 */) noexcept
{
    // Data on the tiles the image will be split into
    unsigned int const tilesHorizontal(8), tilesVertical(8);
    unsigned int const tileWidth(input.cols / tilesHorizontal);
    unsigned int const tileHeight(input.rows / tilesVertical);

    // Make the underlying data of the output the same as the input
    output.create(input.size(), input.type());

    LookupTable * claheLookupTables[tilesVertical][tilesHorizontal];

    for (auto & column : claheLookupTables)
    {
        for (auto & table : column)
        {
            table = new LookupTable;
            assert(nullptr != table);
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
            ImageHistogram tileHistogram;
            generateGrayscaleHistogram(regionOfInterest, tileHistogram);

            // Clip the histogram and redistribute
            clipHistogram(tileHistogram, clipLimit);

            // Perform gray level mapping
            mapping(tileHistogram, claheLookupTables[rowIdx][colIdx]);
        }
    }

    // Now for each pixel, interpolate an intensity value from the gray level mappings of the closest tiles
    for (auto rowIdx = 0u; rowIdx < input.rows; ++rowIdx)
    {
        for (auto colIdx = 0u; colIdx < input.cols; ++colIdx)
        {
            // Find the four closest tile centers and the tile coordinates
            std::array<TileCoordinates, 4> closestTiles{};
            if (isCornerRegion(colIdx, rowIdx, tilesHorizontal, tilesVertical, input, closestTiles))
            {
                // The closest tile is in index 0
                uint8_t currentPixelIntensity = input.at<uint8_t>(rowIdx, colIdx);
                auto currentMappingTable =
                    claheLookupTables[closestTiles[0].y][closestTiles[0].x];
                uint8_t newPixelIntensity =
                    currentMappingTable->operator[](currentPixelIntensity);
                output.at<uint8_t>(rowIdx, colIdx) = newPixelIntensity;
            }
            else if (isBorderRegion(colIdx, rowIdx, tilesHorizontal,
                                    tilesVertical, input, closestTiles))
            {
                // The two closest tiles are indices 0 and 1
                Pixel pixel0 = {
                    // Get the tile center's xy-coordinates
                    getPixelCoordinateFromTileCoordinate(closestTiles[0].x, tileWidth),
                    getPixelCoordinateFromTileCoordinate(closestTiles[0].y, tileHeight),
                    // Plug the input pixel's intensity into the tile's lookup table
                    claheLookupTables[closestTiles[0].y][closestTiles[0].x]->operator[](
                        input.at<uint8_t>(rowIdx, colIdx))};
                Pixel pixel1 = {
                    // Get the tile center's xy-coordinates
                    getPixelCoordinateFromTileCoordinate(closestTiles[1].x, tileWidth),
                    getPixelCoordinateFromTileCoordinate(closestTiles[1].y, tileHeight),
                    // Plug the input pixel's intensity into the tile's lookup table
                    claheLookupTables[closestTiles[1].y][closestTiles[1].x]->operator[](
                        input.at<uint8_t>(rowIdx, colIdx))};
                output.at<uint8_t>(rowIdx, colIdx) = static_cast<uint8_t>(
                    linearInterpolate(pixel0, pixel1, colIdx, rowIdx).intensity);
            }
            else
            {
                // Grab the tile coordinates from all 4 indices
                getFourClosestTiles(colIdx, rowIdx, tilesHorizontal,
                                    tilesVertical, input, closestTiles);
                // Create a pixel for each tile center with an intensity from
                // mapping the current input pixel
                std::vector<Pixel> tileCenters;
                for (auto & tile : closestTiles)
                {
                    tileCenters.push_back(
                        // Get the x-coordinate, y-coordinate of each tile center
                        {getPixelCoordinateFromTileCoordinate(tile.x, tileWidth),
                         getPixelCoordinateFromTileCoordinate(tile.y, tileHeight),
                         // Get the look-up table for this tile and plug this pixel's intensity in
                         claheLookupTables[tile.y][tile.x]->operator[](
                             input.at<uint8_t>(rowIdx, colIdx))});
                }
                // Interpolate between the four pixels and assign it to the output image
                output.at<uint8_t>(rowIdx, colIdx) = static_cast<uint8_t>(
                    bilinearInterpolate(tileCenters, colIdx, rowIdx).intensity);
            }
        }
    }

    // Clean up dynamically allocated memory
    for (auto & column : claheLookupTables)
    {
        for (auto & table : column)
        {
            delete table;
        }
    }

    return 0;
}

static void areaBasedGrayLevelMapping(ImageHistogram const & histogram, LookupTable * outputTable)
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
        outputTable->operator[](i) = static_cast<unsigned char>(
            ratioOfPixelsSeenToTotal * (outputTable->size() - 1));
    }
}

static bool isCornerRegion(unsigned int x,
                           unsigned int y,
                           unsigned int tilesHorizontal,
                           unsigned int tilesVertical,
                           cv::Mat const & input,
                           std::array<TileCoordinates, 4> & outputTile)
{
    unsigned int const tileWidth(input.cols / tilesHorizontal);
    unsigned int const tileHeight(input.rows / tilesVertical);
    // Is it in the top left corner?
    if (x <= (tileWidth / 2) && y <= (tileHeight / 2))
    {
        // Tile coordinate is 0, 0
        outputTile[0] = {0, 0};
        return true;
    }
    // Is it in the top right corner?
    else if (x > ((tileWidth * tilesHorizontal) - tileWidth / 2) && y <= (tileHeight / 2))
    {
        outputTile[0] = {tilesHorizontal - 1, 0};
        return true;
    }
    // Is it in the bottom right corner?
    else if (x >= ((tileWidth * tilesHorizontal) - tileWidth / 2) &&
             y >= ((tileHeight * tilesVertical) - tileHeight / 2))
    {
        outputTile[0] = {tilesHorizontal - 1, tilesVertical - 1};
        return true;
    }
    // Is it in the bottom left corner?
    else if (x <= (tileWidth / 2) && y > ((tileHeight * tilesVertical) - tileHeight / 2))
    {
        outputTile[0] = {0, tilesVertical - 1};
        return true;
    }
    // The pixel is not in a corner
    return false;
}

static bool isBorderRegion(unsigned int x,
                           unsigned int y,
                           unsigned int tilesHorizontal,
                           unsigned int tilesVertical,
                           cv::Mat const & input,
                           std::array<TileCoordinates, 4> & outputTile)
{
    unsigned int const tileWidth(input.cols / tilesHorizontal);
    unsigned int const tileHeight(input.rows / tilesVertical);

    // Is it on the top border?
    if (y <= (tileHeight / 2))
    {
        // Tile coordinates in y direction are 0
        // Now find tile x-coordinates
        unsigned int leftX = getLowerTileCoordinate(x, tileWidth);
        unsigned int rightX = leftX + 1;
        outputTile[0] = {leftX, 0};
        outputTile[1] = {rightX, 0};
        return true;
    }
    // Is it on the bottom border?
    else if (y >= ((tilesVertical * tileHeight) - tileHeight / 2))
    {
        // Tile coordinates in y direction are tilesVertical - 1
        // Now find the x-coordinates
        unsigned int leftX = getLowerTileCoordinate(x, tileWidth);
        unsigned int rightX = leftX + 1;
        outputTile[0] = {leftX, tilesVertical - 1};
        outputTile[1] = {rightX, tilesVertical - 1};
        return true;
    }
    // Is it on the left border?
    else if (x <= (tileWidth / 2))
    {
        // Tile coordinates in x direction are 0
        // Now find the y-coordinates
        unsigned int topY = getLowerTileCoordinate(y, tileHeight);
        unsigned int bottomY = topY + 1;
        outputTile[0] = {0, topY};
        outputTile[1] = {0, bottomY};
        return true;
    }
    // Is it on the right border?
    else if (x >= ((tilesHorizontal * tileWidth) - tileWidth / 2))
    {
        // Tile coordinates in x direction are tilesHorizontal - 1
        // Now find the y-coordinates
        unsigned int topY = getLowerTileCoordinate(y, tileHeight);
        unsigned int bottomY = topY + 1;
        outputTile[0] = {tilesHorizontal - 1, topY};
        outputTile[1] = {tilesHorizontal - 1, bottomY};
        return true;
    }
    // The pixel is not in a border
    return false;
}

static void getFourClosestTiles(unsigned int x,
                                unsigned int y,
                                unsigned int tilesHorizontal,
                                unsigned int tilesVertical,
                                cv::Mat const & input,
                                std::array<TileCoordinates, 4> & outputTile)
{
    unsigned int const tileWidth(input.cols / tilesHorizontal);
    unsigned int const tileHeight(input.rows / tilesVertical);

    unsigned int leftX = getLowerTileCoordinate(x, tileWidth);
    unsigned int rightX = leftX + 1;
    unsigned int topY = getLowerTileCoordinate(y, tileHeight);
    unsigned int bottomY = topY + 1;

    // Top left closest
    outputTile[0] = {leftX, topY};
    // Top right closest
    outputTile[1] = {rightX, topY};
    // Bottom right closest
    outputTile[2] = {rightX, bottomY};
    // Bottom left closest
    outputTile[3] = {leftX, bottomY};
}

static unsigned int getPixelCoordinateFromTileCoordinate(unsigned int tileCoordinate,
                                                         unsigned int pixelsPerTile)
{
    return (pixelsPerTile / 2) + (tileCoordinate * pixelsPerTile);
}

static unsigned int getLowerTileCoordinate(float pixelDimension, float tileDimension)
{
    return static_cast<unsigned int>((pixelDimension - (tileDimension / 2)) / tileDimension);
}

} // namespace snover