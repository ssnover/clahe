/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, NVIDIA Corporation, all rights reserved.
// Copyright (C) 2014, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

// ----------------------------------------------------------------------
// CLAHE

namespace
{
template <class T, int histSize, int shift>
class CLAHE_CalcLut_Body : public cv::ParallelLoopBody
{
public:
    CLAHE_CalcLut_Body(const cv::Mat& src, const cv::Mat& lut, const cv::Size& tileSize, const int& tilesX, const int& clipLimit, const float& lutScale) :
    src_(src), lut_(lut), tileSize_(tileSize), tilesX_(tilesX), clipLimit_(clipLimit), lutScale_(lutScale)
    {
    }

    void operator ()(const cv::Range& range) const override;

private:
    cv::Mat src_;
    mutable cv::Mat lut_;

    cv::Size tileSize_;
    int tilesX_;
    int clipLimit_;
    float lutScale_;
};

template <class T, int histSize, int shift>
void CLAHE_CalcLut_Body<T,histSize,shift>::operator ()(const cv::Range& range) const
{
    T* tileLut = lut_.ptr<T>(range.start);
    // NUmber of bytes in one row of lookup table
    // T is a char
    // So lut_step = number of elements in one row of look up table
    const size_t lut_step = lut_.step / sizeof(T);

    // Iterates over the number of tiles (k: 0 to number of tiles)
    // After each tile, move the pointer forward by number of elements in a single row of lookup table
    // So each row of LUT corresponds to a single tile
    for (int k = range.start; k < range.end; ++k, tileLut += lut_step)
    {
        // Gets the x and y coordinates of the tile number
        const int ty = k / tilesX_;
        const int tx = k % tilesX_;

        // retrieve tile submatrix
        // creates a rectangle describing the tile's pixel coordinates
        cv::Rect tileROI;
        tileROI.x = tx * tileSize_.width;
        tileROI.y = ty * tileSize_.height;
        tileROI.width = tileSize_.width;
        tileROI.height = tileSize_.height;
        // Copies that tile's pixels into a new Mat
        const cv::Mat tile = src_(tileROI);

        // calc histogram

        int tileHist[histSize] = {0, };

        int height = tileROI.height;
        // How many elements per row of original image (add this number to pointer to move down one pixel in image)
        const size_t sstep = src_.step / sizeof(T);
        // For each iteration, do stuff, then move the pointer down one pixel in y
        // Each loop iterates on a row of the tile
        for (const T* ptr = tile.ptr<T>(0); height--; ptr += sstep)
        {
            int x = 0;
            // Iterate over the tile row, adding four pixels at a time to the histogram
            for (; x <= tileROI.width - 4; x += 4)
            {
                int t0 = ptr[x], t1 = ptr[x+1];
                tileHist[t0 >> shift]++; tileHist[t1 >> shift]++;
                t0 = ptr[x+2]; t1 = ptr[x+3];
                tileHist[t0 >> shift]++; tileHist[t1 >> shift]++;
            }
            // This appears to be redundant. Was written by jet47 at the same time as the above loop
            for (; x < tileROI.width; ++x)
                tileHist[ptr[x] >> shift]++;
        }

        // clip histogram

        if (clipLimit_ > 0)
        {
            // how many pixels were clipped
            int clipped = 0;
            // iterate over each bin in the histogram
            for (int i = 0; i < histSize; ++i)
            {
                // if the height for this bin is greater than limit
                if (tileHist[i] > clipLimit_)
                {
                    // subtract those pixels above limit
                    clipped += tileHist[i] - clipLimit_;
                    // add those pixels to the pool of clipped pixels
                    tileHist[i] = clipLimit_;
                }
            }

            // redistribute clipped pixels

            // number of clipped pixels that will be added back to each bin
            int redistBatch = clipped / histSize;
            // left over, could be clipped % histSize
            int residual = clipped - redistBatch * histSize;

            // add those pixels back to the histogram evenly
            for (int i = 0; i < histSize; ++i)
                tileHist[i] += redistBatch;

            if (residual != 0)
            {
                // add the residual to the lowest intensity bins first
                int residualStep = MAX(histSize / residual, 1);
                for (int i = 0; i < histSize && residual > 0; i += residualStep, residual--)
                    tileHist[i]++;
            }
        }

        // calc Lut

        // holds the number of pixels in the histogram
        int sum = 0;
        // for each bin of the histogram
        for (int i = 0; i < histSize; ++i)
        {
            sum += tileHist[i];
            // for this tile's look up table, set this bin's value
            // sum is total number of pixels in this bin and to the left
            // redistributes bins such that CDF is closer to one
            // if an image is very bright, and at the histSize / 2, only one quarter of the pixels have been seen
            // then the tileLut will get histSize / 4. This denotes where these pixels should be for a normalized histogram with CDF of 1
            tileLut[i] = cv::saturate_cast<T>(sum * lutScale_);
        }
    }
}

template <class T, int shift>
class CLAHE_Interpolation_Body : public cv::ParallelLoopBody
{
public:
    CLAHE_Interpolation_Body(const cv::Mat& src, const cv::Mat& dst, const cv::Mat& lut, const cv::Size& tileSize, const int& tilesX, const int& tilesY) :
    src_(src), dst_(dst), lut_(lut), tileSize_(tileSize), tilesX_(tilesX), tilesY_(tilesY)
    {
        buf.allocate(src.cols << 2);
        // Points at start of buffer
        ind1_p = buf.data();
        // Points a quarter way through the buffer
        ind2_p = ind1_p + src.cols;
        // Points halfway through buffer, and at a float
        xa_p = (float *)(ind2_p + src.cols);
        // Points three quarters through buffer, at a float
        xa1_p = xa_p + src.cols;

        // figures out number of elements per row of lookup table
        int lut_step = static_cast<int>(lut_.step / sizeof(T));

        float tileWidthInverse = 1.0f / tileSize_.width;

        // For each column
        for (int x = 0; x < src.cols; ++x)
        {
            // finds which tile region each column goes in
            float txf = x * tileWidthInverse - 0.5f;
            // finds the tileX on each side of each column
            // each column determines if its on left or right half of tileX
            // if left half, it will be considered in the interpolation of previous tile and current tile
            // if right half, it will be considered in interpolation of current tile and next tile
            // tx1 is the closest tileX to the left
            int tx1 = cvFloor(txf);
            // tx2 is the closest tileX to the right
            int tx2 = tx1 + 1;

            xa_p[x] = txf - tx1;
            xa1_p[x] = 1.0f - xa_p[x];

            tx1 = std::max(tx1, 0);
            tx2 = std::min(tx2, tilesX_ - 1);

            ind1_p[x] = tx1 * lut_step;
            ind2_p[x] = tx2 * lut_step;
        }
    }

    void operator ()(const cv::Range& range) const override;

private:
    cv::Mat src_;
    mutable cv::Mat dst_;
    cv::Mat lut_;

    cv::Size tileSize_;
    int tilesX_;
    int tilesY_;

    cv::AutoBuffer<int> buf;
    int * ind1_p, * ind2_p;
    float * xa_p, * xa1_p;
};

template <class T, int shift>
void CLAHE_Interpolation_Body<T, shift>::operator ()(const cv::Range& range) const
{
    // Basically, this finds the four closest tile center coordinates
    // Then it applies the gray level mapping for each of these tiles
    // It then runs bilinear interpolation where the pixel coordinates are the tile centers and the intensity values
    // are the output of the gray level mapping in each tile
    float inv_th = 1.0f / tileSize_.height;

    // For each row
    for (int y = range.start; y < range.end; ++y)
    {
        const T* srcRow = src_.ptr<T>(y);
        T* dstRow = dst_.ptr<T>(y);

        float tyf = y * inv_th - 0.5f;

        int ty1 = cvFloor(tyf);
        int ty2 = ty1 + 1;

        float ya = tyf - ty1, ya1 = 1.0f - ya;

        ty1 = std::max(ty1, 0);
        ty2 = std::min(ty2, tilesY_ - 1);

        const T* lutPlane1 = lut_.ptr<T>(ty1 * tilesX_);
        const T* lutPlane2 = lut_.ptr<T>(ty2 * tilesX_);

        for (int x = 0; x < src_.cols; ++x)
        {
            int srcVal = srcRow[x] >> shift;

            int ind1 = ind1_p[x] + srcVal;
            int ind2 = ind2_p[x] + srcVal;

            float res = (lutPlane1[ind1] * xa1_p[x] + lutPlane1[ind2] * xa_p[x]) * ya1 +
                        (lutPlane2[ind1] * xa1_p[x] + lutPlane2[ind2] * xa_p[x]) * ya;

            dstRow[x] = cv::saturate_cast<T>(res) << shift;
        }
    }
}

class CLAHE_Impl final : public cv::CLAHE
{
public:
    CLAHE_Impl(double clipLimit = 40.0, int tilesX = 8, int tilesY = 8);

    void apply(cv::InputArray src, cv::OutputArray dst) override;

    void setClipLimit(double clipLimit) override;
    double getClipLimit() const override;

    void setTilesGridSize(cv::Size tileGridSize) override;
    cv::Size getTilesGridSize() const override;

    void collectGarbage() override;

private:
double clipLimit_;
int tilesX_;
int tilesY_;

cv::Mat srcExt_;
cv::Mat lut_;
};

CLAHE_Impl::CLAHE_Impl(double clipLimit, int tilesX, int tilesY) :
clipLimit_(clipLimit), tilesX_(tilesX), tilesY_(tilesY)
{
}

void CLAHE_Impl::apply(cv::InputArray _src, cv::OutputArray _dst)
{
    CV_INSTRUMENT_REGION();

    // Make sure the source image is the right type
    CV_Assert( _src.type() == CV_8UC1 || _src.type() == CV_16UC1 );

    // Match the histogram size to the type
    int histSize = _src.type() == CV_8UC1 ? 256 : 65536;

    cv::Size tileSize;
    cv::_InputArray _srcForLut;

    // Do the tiles fit perfectly in the image?
    if (_src.size().width % tilesX_ == 0 && _src.size().height % tilesY_ == 0)
    {
        // Yes, get the tile size and assign the source
        tileSize = cv::Size(_src.size().width / tilesX_, _src.size().height / tilesY_);
        _srcForLut = _src;
    }
    else
    {
        // No, pad the bottom and right side to remove remainder with reflected pixels
        cv::copyMakeBorder(_src, srcExt_, 0, tilesY_ - (_src.size().height % tilesY_), 0, tilesX_ - (_src.size().width % tilesX_), cv::BORDER_REFLECT_101);
        tileSize = cv::Size(srcExt_.size().width / tilesX_, srcExt_.size().height / tilesY_);
        _srcForLut = srcExt_;
    }

    // Get the area of each tile
    const int tileSizeTotal = tileSize.area();
    // Calculate a normalizing constant from area for LUT
    const float lutScale = static_cast<float>(histSize - 1) / tileSizeTotal;

    int clipLimit = 0;
    if (clipLimit_ > 0.0)
    {
        // Generate a clip limit from parameters
        clipLimit = static_cast<int>(clipLimit_ * tileSizeTotal / histSize);
        // Make sure the clip limit is a positive integer, if not make it 1
        clipLimit = std::max(clipLimit, 1);
    }
    // Get the underlying data representation and make a matching one for destination
    cv::Mat src = _src.getMat();
    _dst.create( src.size(), src.type() );
    cv::Mat dst = _dst.getMat();
    cv::Mat srcForLut = _srcForLut.getMat();
    // Create the look up table with the number of tiles
    // One dimension is the tile number
    // Second dimension is the number of bins in the histogram
    lut_.create(tilesX_ * tilesY_, histSize, _src.type());

    cv::Ptr<cv::ParallelLoopBody> calcLutBody;

    // Fill the look up table
    if (_src.type() == CV_8UC1)
        calcLutBody = cv::makePtr<CLAHE_CalcLut_Body<uchar, 256, 0> >(srcForLut, lut_, tileSize, tilesX_, clipLimit, lutScale);
    else if (_src.type() == CV_16UC1)
        calcLutBody = cv::makePtr<CLAHE_CalcLut_Body<ushort, 65536, 0> >(srcForLut, lut_, tileSize, tilesX_, clipLimit, lutScale);
    else
        CV_Error( CV_StsBadArg, "Unsupported type" );

    // Executes calcLutBody operator()
    cv::parallel_for_(cv::Range(0, tilesX_ * tilesY_), *calcLutBody);

    cv::Ptr<cv::ParallelLoopBody> interpolationBody;
    if (_src.type() == CV_8UC1)
        interpolationBody = cv::makePtr<CLAHE_Interpolation_Body<uchar, 0> >(src, dst, lut_, tileSize, tilesX_, tilesY_);
    else if (_src.type() == CV_16UC1)
        interpolationBody = cv::makePtr<CLAHE_Interpolation_Body<ushort, 0> >(src, dst, lut_, tileSize, tilesX_, tilesY_);
    // Executes the interopolation operator()
    cv::parallel_for_(cv::Range(0, src.rows), *interpolationBody);
}

void CLAHE_Impl::setClipLimit(double clipLimit)
{
    clipLimit_ = clipLimit;
}

double CLAHE_Impl::getClipLimit() const
{
    return clipLimit_;
}

void CLAHE_Impl::setTilesGridSize(cv::Size tileGridSize)
{
    tilesX_ = tileGridSize.width;
    tilesY_ = tileGridSize.height;
}

cv::Size CLAHE_Impl::getTilesGridSize() const
{
    return cv::Size(tilesX_, tilesY_);
}

void CLAHE_Impl::collectGarbage()
{
    srcExt_.release();
    lut_.release();
}
} // namespace

cv::Ptr<cv::CLAHE> cv::createCLAHE(double clipLimit, cv::Size tileGridSize)
{
    return makePtr<CLAHE_Impl>(clipLimit, tileGridSize.width, tileGridSize.height);
}