#include "computeCoeffDCT.h"

#include <Halide.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace Halide;

void computeCoeffDCTFromNormalKernel(int dcttype, int radius, int order, std::function<double(int, double, int)> kernel, vector<double>& Gk)
{
	double T = 0.0;
	double k0 = 0.0;
	double r0 = 0.0;
	switch (dcttype)
	{
	case 1:	T = 2.0 * radius + 0.0; k0 = 0.0; r0 = 0.0; break;
	case 3:	T = 2.0 * radius + 2.0; k0 = 0.5; r0 = 0.0; break;
	case 5:	T = 2.0 * radius + 1.0; k0 = 0.0; r0 = 0.0; break;
	case 7:	T = 2.0 * radius + 1.0; k0 = 0.5; r0 = 0.0; break;
	default: throw "Unsupported DCT type"; break;
	}

	double omega = CV_2PI / T;

	for (int k = 0; k <= order; k++)
	{
		double sum = 0.0;
		for (int r = -radius; r <= radius; r++)
		{
			sum += kernel(radius, r0, r) * cos(omega * (k + k0) * (r + r0));
		}

		Gk[k] = (2.0 / T) * sum;
	}

	if (dcttype == 1 || dcttype == 2 || dcttype == 5 || dcttype == 6)  Gk[0] *= 0.5;

}

bool computeAndOptimizeCoeffDCTFromNormalKernel(const int dcttype, const int radius, const int order, std::function<double(int, double, int)> kernel, vector<double>& Gk)
{
	double omega, k0, T, n0 = 0.0;
	switch (dcttype)
	{
	case 1: T = (2.0 * radius + 0.0); omega = CV_2PI / (2.0 * radius + 0.0); k0 = 0.0; break;
	case 3: T = (2.0 * radius + 2.0); omega = CV_2PI / (2.0 * radius + 2.0); k0 = 0.5; break;
	case 5: T = (2.0 * radius + 1.0); omega = CV_2PI / (2.0 * radius + 1.0); k0 = 0.0; break;
	case 7:	T = (2.0 * radius + 1.0); omega = CV_2PI / (2.0 * radius + 1.0); k0 = 0.5; break;
	default: throw "Unsupported DCT type"; break;
	}

	//kernel
	cv::Mat1d h0(radius + 1, 1);

	for (int r = 0; r <= radius; r++)
	{
		h0.at<double>(r, 0) = kernel(radius, n0, r);
	}

	//weight matrix
	cv::Mat1d W = cv::Mat1d::eye(radius + 1, radius + 1);
	W(0, 0) = 0.5;

	// DCT matrix
	cv::Mat1d C(radius + 1, order + 1);
	for (int n = 0; n <= radius; ++n)
	{
		for (int k = 0; k <= order; ++k)
		{
			C(n, k) = cos(omega * (k + k0) * (n + n0));
		}
	}

	cv::Mat1d CWCinv;//K+1 x K+1
	cv::Mat1d a_ls;

	//kernel
	cv::Mat1d h0Full(2 * radius + 1, 1);

	for (int r = -radius; r <= radius; r++)
	{
		h0Full.at<double>(r + radius, 0) = kernel(radius, n0, r);
	}

	// DCT matrix
	cv::Mat1d CFull(2 * radius + 1, order + 1);
	for (int n = -radius; n <= radius; ++n)
	{
		for (int k = 0; k <= order; ++k)
		{
			CFull(n + radius, k) = cos(omega * (k + k0) * (n + n0));
		}
	}

	CWCinv = (CFull.t() * CFull).inv(DECOMP_CHOLESKY);
	a_ls = CWCinv * CFull.t() * h0Full;

	if (a_ls(0, 0) == 0)
	{
		computeCoeffDCTFromNormalKernel(dcttype, radius, order, kernel, Gk);
		return false;
	}

	for (int k = 0; k <= order; ++k)
	{
		Gk[k] = a_ls(k, 0);
	}

	return true;
}

