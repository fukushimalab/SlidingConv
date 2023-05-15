# SlidingConv

A Simple and Fast DSL for Sliding-DCT.

## Requirements
- OpenCV 4.5.0
- Halide 12.0.1
- C++ 14
- Visual Studio 2019 or 2022
- Windows 10 or 11

You can use a pre-built binary of [OpenCV](https://github.com/opencv/opencv/releases) and [Halide](https://github.com/halide/Halide/releases), but if you want to test with a CUDA target, you need to build Halide from the source code yourself.

## Setup

You need to set following Environment Variables.

- `OPENCV_INCLUDE_DIR`: A path to the `include` directory of OpenCV.
- `OPENCV_LIB_DIR`: A path to the `lib` directory of OpenCV.
- `HALIDE_INCLUDE_DIR`: A path to the `include` directory of Halide.
- `HALIDE_LIB_DIR`: A path to the `lib` directory of Halide.

You also need to add the `bin` direcotries of OpenCV and Halide to the PATH.

## Demo

`main.cpp` has 2 demos. You need to add a `.png` or `.jpg` image to the root directory and write its filename to main.cpp.

- `gaussian`: Gaussian filter
- `unsharp`: Unsharpmasking
