# Contrast-Limited Adaptive Histogram Equalization (CLAHE)
CLAHE is a technique for increasing image contrast in order to view finer detail by which the histogram is used to determine the best way to alter the image to increase contrast.

CLAHE is commonly used in fields including biomedical imaging and electronics manufacturing to aid in detection of various features. It has many implementations and small tweaks have been made to optimize it for various applications.

I am implementing a CLAHE algorithm in C++17 for my computer vision term project at RIT for CSCI 631 Foundations of Computer Vision. The goal of the project is to produce an implementation of CLAHE and evaluate it on a set of images by comparing its performance and output to results produced by other available implementations like the one produced by OpenCV and Mathworks MATLAB's Image Processing Toolbox.