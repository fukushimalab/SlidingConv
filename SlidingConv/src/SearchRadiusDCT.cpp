#include "SearchRadiusDCT.h"

#include "computeCoeffDCT.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

double SearchRadiusDCT::getError(int x)
{
	double T = 0.0;
	double k0 = 0.0;
	double r0 = 0.0;
	switch (dctType)
	{
	case 1:	T = 2.0 * x + 0.0; k0 = 0.0; r0 = 0.0; break;
	case 3:	T = 2.0 * x + 2.0; k0 = 0.5; r0 = 0.0; break;
	case 5:	T = 2.0 * x + 1.0; k0 = 0.0; r0 = 0.0; break;
	case 7:	T = 2.0 * x + 1.0; k0 = 0.5; r0 = 0.0; break;
	default: throw "Unsupported DCT type"; break;
	}

	double omega = CV_2PI / T;

	double Es;
	{
		double sumLeft = 0.0;
		for (int r = -infiniteR; r <= -(x + 1); r++)
		{
			double hr = kernel(infiniteR, r0, r);
			sumLeft += hr * hr;
		}

		double sumRight = 0.0;
		for (int r = x + 1; r <= infiniteR; r++)
		{
			double hr = kernel(infiniteR, r0, r);
			sumRight += hr * hr;
		}

		Es = (sumLeft + sumRight) / sum;
	}

	double Ef = 0.0;
	{
		vector<double> Gk(order + 1);
		if (coeffOptimize) computeAndOptimizeCoeffDCTFromNormalKernel(dctType, x, order, kernel, Gk);
		else computeCoeffDCTFromNormalKernel(dctType, x, order, kernel, Gk);

		for (int r = -x; r <= x; r++)
		{
			double ideal = kernel(x, r0, r);
			double result = 0.0;
			for (int k = 0; k <= order; k++)
			{
				result += Gk[k] * cos(omega * (k + k0) * (r + r0));
			}

			Ef += (ideal - result) * (ideal - result);
		}

		Ef /= sum;
	}

	return Es + Ef;
}

SearchRadiusDCT::SearchRadiusDCT(int dctType, int order, std::function<double(int, double, int)> kernel, bool coeffOptimize) : dctType(dctType), order(order), kernel(kernel), coeffOptimize(coeffOptimize)
{
	double r0 = 0.0;
	if (dctType % 2 == 0) r0 = 0.5;

	sum = 0.0;
	for (int r = -infiniteR; r <= infiniteR; r++)
	{
		double hr = kernel(infiniteR, r0, r);
		sum += hr * hr;
	}
}