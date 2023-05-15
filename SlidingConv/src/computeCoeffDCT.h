#pragma once
#include <functional>
#include <vector>

void computeCoeffDCTFromNormalKernel(int dcttype, int radius, int order, std::function<double(int, double, int)> kernel, std::vector<double>& Gk);

bool computeAndOptimizeCoeffDCTFromNormalKernel(const int dcttype, const int radius, const int order, std::function<double(int, double, int)> kernel, std::vector<double>& Gk);