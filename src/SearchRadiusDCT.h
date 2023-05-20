#pragma once

#include <functional>
#include "search1D.hpp"

class SearchRadiusDCT : public util::Search1DInt
{
private:
	int dctType;

	int order;
	std::function<double(int, double, int) > kernel;
	bool coeffOptimize;

	// ∞の代わりに使う値
	int infiniteR = 100;
	double sum;

public:
	double getError(int x);

	SearchRadiusDCT(int dctType, int order, std::function<double(int, double, int)> kernel, bool coeffOptimize);
};

