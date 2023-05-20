# SlidingConv

A Simple and Fast DSL for Sliding-DCT.

[View on GitHub](https://github.com/fukushimalab/SlidingConv)

# Demo

`demo.cpp` is a demo for SlidingConv. It has 2 filters `Gaussian Filter` and `Unsharpmasking`.

We **cofirmed** that SlidingCov is working on following combination. (Other combination may be work but not be confirmed.)

| Compiler | OS | Arch |
| ----- | ----- | ----- |
| MSVC/14.29.30133 (MSVC 2019) | Windows 11 Home | x86/64 |
| Apple clang 14.0.3 (Xcode 14.3.0) | macOS Ventura 13.3.1 (22E261) | arm64 |
| g++ 9.4.0 | Ubuntu 20.04.6 LTS | x86/64 |

## Requirements
- OpenCV 4.5.0
- Halide 12.0.1
    - Halide requires `zlib`, `libpng`, `libjpeg`

You can use a pre-built binary of [Halide](https://github.com/halide/Halide/releases), but if you want to test with a GPU (CUDA target), you need to build Halide from the source code yourself.

## Setup

You need to add the Path of `Halide` and `OpenCV` to `CMAKE_PREFIX_PATH` before cmake. (or set `Halide_DIR` and `OpenCV_DIR` directly.)

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
    - 1=Run on GPU (CUDA Target)
- sigma: sigma for gaussian kernel

e.g.
```sh
./SlidingConvDemo image.png 0 3.0
```

# LICENSE

MIT

- [recfilter](https://github.com/mit-gfx/recfilter/blob/master/LICENSE.txt)
- [Halide](https://github.com/halide/Halide/blob/main/LICENSE.txt)
- [OpenCV](https://github.com/opencv/opencv/blob/4.x/LICENSE)
