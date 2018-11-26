cmake_minimum_required(VERSION 3.10)
project(clahe)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR $ENV{OPENCV_PATH})
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)

add_executable(clahe main.cpp
                     clahe.hpp
        )
target_link_libraries(clahe ${OpenCV_LIBS})
target_include_directories(clahe PUBLIC ${OpenCV_INCLUDE_DIRS})

add_executable(opencv-clahe opencv-clahe.cpp)
target_link_libraries(opencv-clahe ${OpenCV_LIBS})
target_include_directories(opencv-clahe PUBLIC ${OpenCV_INCLUDE_DIRS})