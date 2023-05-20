#include <iostream>
#include <opencv2/opencv.hpp>
#include "util/halide_benchmark.h"
#include "util/HalideWithOpenCV.h"
#include "src/SlidingConv.h"

using namespace std;
using namespace cv;
using namespace Halide;

void gaussian(string filename, double sigma, bool isGPU) {
	Mat cv_input = imread(filename, 0), cv_input_float;
	cv_input.convertTo(cv_input_float, CV_32F, 1 / 255.f);
	Mat normal_output(cv_input_float.size(), cv_input_float.type());
	Mat halide_output(cv_input_float.size(), cv_input_float.type());

	int w = cv_input.cols;
	int h = cv_input.rows;
	int d = cv_input.channels();
	Buffer<float> halide_input_float(w, h, d);
	convMat2Halide(cv_input_float, halide_input_float);

	// gaussian kernel
	function<double(int, double, int)> kernel = [&](int maxRadius, double offset, int r)
	{
		double sum = 0.0;
		for (int i = -maxRadius; i <= maxRadius; i++)
		{
			double offseted = i + offset;
			sum += exp(-(offseted * offseted) / (2 * sigma * sigma));
		}

		double offseted = r + offset;
		double ex = exp(-(offseted * offseted) / (2 * sigma * sigma));

		return ex / sum;
	};

	Var x("x"), y("y"), c("c");
	SlidingConv gf("gf", w, h, d);

	gf(x, y, c) = halide_input_float(x, y, c);

	gf
		.set_kernel(kernel)
		.set_order(3)
		.set_radius(0)
		.set_optimizeCoeff(true)
		.set_algorithm(SlidingAlgorithm::DCT5);

	if (isGPU)
	{
		gf.set_target_feature(Target::CUDA);
		gf.gpu_auto_schedule();
	}
	else
	{
		gf.cpu_auto_schedule();
	}

	// you have to call this before realize()
	gf.prepareForRealize();

	Buffer<float> final_out(w, h, d);

	double result = Tools::benchmark(100, 100, [&]()
		{
			gf.realize(final_out);
		}
	);

	if (isGPU)
	{
		final_out.copy_to_host();
	}
	convHalide2Mat(final_out, halide_output);

	// answer image
	int cvRadius = 6 * sigma;
	GaussianBlur(cv_input_float, normal_output, Size(2 * cvRadius + 1, 2 * cvRadius + 1), sigma);
	double psnr = PSNR(normal_output, halide_output, 1.0);

	cout << "Gaussian time: " << result * 1e3 << " ms PSNR: " << psnr << " dB" << endl;

	Mat halide_output_uchar, normal_output_uchar;
	halide_output.convertTo(halide_output_uchar, CV_8U, 255.f);
	normal_output.convertTo(normal_output_uchar, CV_8U, 255.f);

	imwrite("Gaussian_SlidingConv.png", halide_output_uchar);
	imwrite("Gaussian_OpenCV.png", normal_output_uchar);
}

void unsharp(string filename, double sigma, bool isGPU) {
	Mat cv_input = imread(filename, 0), cv_input_float;
	cv_input.convertTo(cv_input_float, CV_32F, 1 / 255.f);
	Mat normal_output(cv_input_float.size(), cv_input_float.type());
	Mat halide_output(cv_input_float.size(), cv_input_float.type());

	int w = cv_input.cols;
	int h = cv_input.rows;
	int d = cv_input.channels();
	Buffer<uchar> halide_input(w, h, d);
	convMat2Halide(cv_input, halide_input);

	// gaussian kernel
	function<double(int, double, int)> kernel = [&](int maxRadius, double offset, int r)
	{
		double sum = 0.0;
		for (int i = -maxRadius; i <= maxRadius; i++)
		{
			double offseted = i + offset;
			sum += exp(-(offseted * offseted) / (2 * sigma * sigma));
		}

		double offseted = r + offset;
		double ex = exp(-(offseted * offseted) / (2 * sigma * sigma));

		return ex / sum;
	};

	Var x("x"), y("y"), c("c");
	SlidingConv gf("gf", w, h, d);

	gf(x, y, c) = cast<float>(halide_input(x, y, c)) / 255.f;

	gf
		.set_kernel(kernel)
		.set_order(3)
		.set_radius(0)
		.set_optimizeCoeff(true)
		.set_algorithm(SlidingAlgorithm::DCT5);

	Func hx("hx");
	hx(x, y, c) = undef<float>();
	hx(x, y, c) = halide_input(x, y, c) / 255.f + 2.f * (halide_input(x, y, c) / 255.f - hx(x, y, c));
	gf.set_post_process_func(hx);

	if (isGPU)
	{
		gf.set_target_feature(Target::CUDA);
		gf.gpu_auto_schedule();
	}
	else
	{
		gf.cpu_auto_schedule();
	}

	// you have to call this before realize()
	gf.prepareForRealize();

	Buffer<float> final_out(w, h, d);

	double result = Tools::benchmark(100, 100, [&]()
		{
			gf.realize(final_out);
		}
	);

	if (isGPU)
	{
		final_out.copy_to_host();
	}
	convHalide2Mat(final_out, halide_output);

	// answer image
	int cvRadius = 6 * sigma;
	GaussianBlur(cv_input_float, normal_output, Size(2 * cvRadius + 1, 2 * cvRadius + 1), sigma);
	normal_output = cv_input_float + 2.f * (cv_input_float - normal_output);
	double psnr = PSNR(normal_output, halide_output, 1.0);

	cout << "Unsharp time: " << result * 1e3 << " ms PSNR: " << psnr << " dB" << endl;

	Mat halide_output_uchar, normal_output_uchar;
	halide_output.convertTo(halide_output_uchar, CV_8U, 255.f);
	normal_output.convertTo(normal_output_uchar, CV_8U, 255.f);

	imwrite("Unsharp_SlidingConv.png", halide_output_uchar);
	imwrite("Unsharp_OpenCV.png", normal_output_uchar);
}

int main(int argc, char* argv[])
{
	if (argc != 4) {
        std::cerr << "usage: ./SlidingConvDemo filename isGPU sigma" << std::endl;
		std::cerr << "filename: A path to the filename of input image" << std::endl;
		std::cerr << "isGPU: 0=false 1=true" << std::endl;
		std::cerr << "sigma: sigma for gaussian kernel" << std::endl;
		std::cerr << "e.g.,: ./SlidingConvDemo image.png 0 3.0" << std::endl;
        return 1;
    }
	string filename = argv[1];
	bool isGPU = static_cast<bool>(std::stoi(argv[2]));
	double sigma = stod(argv[3]);

	gaussian(filename, sigma, isGPU);
	unsharp(filename, sigma, isGPU);
}