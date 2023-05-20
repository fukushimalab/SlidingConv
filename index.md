# SlidingConv

A Simple and Fast DSL for Sliding-DCT.

[View GitHub](https://github.com/fukushimalab/SlidingConv)

# Demo

`demo.cpp` is a demo for SlidingConv. It has 2 filters `Gaussian Filter` and `Unsharpmasking`.

## Requirements
- OpenCV 4.5.0
- Halide 12.0.1
    - Halide requires `zlib`, `libpng`, `libjpeg`

You can use a pre-built binary of [OpenCV](https://github.com/opencv/opencv/releases) and [Halide](https://github.com/halide/Halide/releases), but if you want to test with a CUDA target, you need to build Halide from the source code yourself.

## Setup

You need to add the Path of Requirements to `CMAKE_PREFIX_PATH` before cmake. (or set `Halide_DIR` and `OpenCV_DIR` directly.)

```sh
cd path/to/root
mkdir build
cd build
cmake ..
make -j8
```

## Run

```sh
./SlidingConvDemo filename isGPU sigma
```

- filename: A path to the filename of input image
- isGPU: Run on GPU or not
    - 0=Run on CPU 
    - 1=Run on GPU
- sigma: sigma for gaussian kernel

e.g.,

```sh
./SlidingConvDemo image.png 0 3.0
```
