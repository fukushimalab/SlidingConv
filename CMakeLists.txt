cmake_minimum_required(VERSION 3.8)
project(SlidingConv VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set paths to the libraries and includes
find_package(OpenCV 4.5.0 REQUIRED)
find_package(Halide 12.0.1 REQUIRED)

# Add the source files
set(LIB_SOURCES
    recfilter/coefficients.cpp
    recfilter/iir_coeff.cpp
    recfilter/modifiers.cpp
    recfilter/recfilter.cpp
    recfilter/recfilter_utils.cpp
    recfilter/reorder.cpp
    recfilter/schedule.cpp
    recfilter/split.cpp
    recfilter/split_custom_sliding_single.cpp
    recfilter/split_custom_sliding_single_multi_delta.cpp
    recfilter/split_custom_sliding_single_multi_delta_post_process.cpp
    recfilter/split_custom_sliding_single_post_process.cpp
    recfilter/timing.cpp
    src/computeCoeffDCT.cpp
    src/computeSlidingFunc.cpp
    src/search1D.hpp
    src/SearchRadiusDCT.cpp
    src/SlidingConvGetters.cpp
    src/SlidingConvInput.cpp
    src/SlidingConvOutput.cpp
    src/SlidingConvRefExpr.cpp
    src/SlidingConvRefSchedule.cpp
    src/SlidingConvRefVar.cpp
    src/SlidingConvSchedule.cpp
    src/SlidingConvSetters.cpp
)

# Define the library
add_library(SlidingConvLib STATIC ${LIB_SOURCES})
target_link_libraries(SlidingConvLib PRIVATE Halide::Halide ${OpenCV_LIBS})

# Define the executable
add_executable(SlidingConvDemo demo.cpp)
target_link_libraries(SlidingConvDemo PRIVATE SlidingConvLib ${OpenCV_LIBS} Halide::Halide)