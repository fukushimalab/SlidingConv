#pragma once

#include <Halide.h>
#include <opencv2/opencv.hpp>

template <typename T>
void convHalide2Mat(const Halide::Buffer<T>& src, cv::Mat& dest) {
	Halide::Buffer<T> cp = src.copy();
	//srcがset_minでオフセッ指定されてると参照エラーの可能性アリ
	cp.set_min(0, 0);
	dest.create(cv::Size(cp.width(), cp.height()), CV_MAKETYPE(cv::DataType<T>::type, cp.channels()));
	T* ptr = dest.ptr<T>(0);
	switch (dest.channels()) {
	case 1:
	{
		for (int y = 0; y < dest.rows; y++) {
			for (int x = 0; x < dest.cols; x++) {
				*(ptr++) = cp(x, y);
			}
		}
		break;
	}
	case 3:
	{
		for (int y = 0; y < dest.rows; y++) {
			for (int x = 0; x < dest.cols; x++) {
				*(ptr++) = cp(x, y, 0);
				*(ptr++) = cp(x, y, 1);
				*(ptr++) = cp(x, y, 2);
			}
		}
		break;
	}
	}

}

template <typename T>
void convMat2Halide(const cv::Mat& src, Halide::Buffer<T>& dest) {
	const T* ptr = src.ptr<T>(0);
	switch (src.channels()) {
	case 1:
	{
		for (int y = 0; y < src.rows; y++) {
			for (int x = 0; x < src.cols; x++) {
				dest(x, y) = *(ptr++);
			}
		}
		break;

	}
	case 3:
	{
		for (int y = 0; y < src.rows; y++) {
			for (int x = 0; x < src.cols; x++) {
				dest(x, y, 0) = *(ptr++);
				dest(x, y, 1) = *(ptr++);
				dest(x, y, 2) = *(ptr++);
			}
		}
		break;
	}
	}
}


// example : Buffer<uint8_t> inputBuff = Halide_imread<uint8_t>(filename, true);
// This maybe does not support float type.
// If you want to read as float, input as uchar and then covert to float.
template <typename T>
Halide::Buffer<T> imread(const std::string fname, bool flg) {
	cv::Mat temp = cv::imread(fname, flg);
	if (temp.empty())
		std::cerr << fname << "is not available" << std::endl;
	Halide::Buffer<T> ret(temp.cols, temp.rows, temp.channels());
	convMat2Halide(temp, ret);
	return ret;
}

template <typename T>
void imwrite(const std::string fname, const Halide::Buffer<T>& src) {
	cv::Mat temp(cv::Size(src.width(), src.height()), CV_MAKETYPE(cv::DataType<T>::type, src.channels()));
	convHalide2Mat(src, temp);
	cv::imwrite(fname, temp);
}


template <typename T>
void Halide_imshow(const std::string wname, Halide::Buffer<T> src) {
	cv::Mat temp;
	convHalide2Mat(src, temp);
	cv::imshow(wname, temp);
}