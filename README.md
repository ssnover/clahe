# Contrast-Limited Adaptive Histogram Equalization (CLAHE)
CLAHE is a technique for increasing image contrast in order to view finer detail by which the histogram is used to determine the best way to alter the image to increase contrast.

CLAHE is commonly used in fields including biomedical imaging and electronics manufacturing to aid in detection of various features. It has many implementations and small tweaks have been made to optimize it for various applications.

I am implementing a CLAHE algorithm in C++17 for my computer vision term project at RIT for CSCI 631 Foundations of Computer Vision. The goal of the project is to produce an implementation of CLAHE and evaluate it on a set of images by comparing its performance and output to results produced by other available implementations like the one produced by OpenCV and Mathworks MATLAB's Image Processing Toolbox.

## Building the Code
### Requirements
* CMake 3.10+ (this could very likely be 3.0, but I haven't tested)
* OpenCV v3.4+ (again, could be as low as 3.0, not using any advanced or new features)
* Clang or GCC version which supports C++17.

### Commands
For debug builds:
```
mkdir Debug && cd Debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
make
```

For release builds:
```
mkdir Release && cd Release
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG"
make
```

## Future Work
* A number of other gray level mappings are possible and it'd be nice to have a header which contains many common ones as functions, at least as examples. There is a single example of passing a function in for a "unity" mapping which should return the input image without alterations.
* Classifying the regions of the image (as corners, borders, interiors) could be done more optimally since they are well-defined by their pixel boundaries.
* Support for color images by converting to YCbCr and performing the function on the Y-channel before merging it and converting back to RGB.
* Rewrite my paper in LaTeX so I can put source on here instead of a PDF.
* Maybe make it possible to run at compile time as a fun experiment.
* Removing dependency on all of OpenCV, would be nice to have a library just for image encoding and decoding and representing the image conveniently.