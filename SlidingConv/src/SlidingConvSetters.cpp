#include "SlidingConv.h"

SlidingConv::SlidingConv(std::string name, int width, int height, int channels)
{
	this->content = SlidingConvContent(name, width, height, channels);
}

SlidingConv& SlidingConv::set_kernel(std::function<double(int, double, int)> kernel)
{
	this->content.kernel = kernel;
	return *this;
}

SlidingConv& SlidingConv::set_order(int order)
{
	this->content.order = order;
	return *this;
}

SlidingConv& SlidingConv::set_radius(int radius)
{
	this->content.radius = radius;
	return *this;
}

SlidingConv& SlidingConv::set_optimizeCoeff(bool optimizeCoeff)
{
	this->content.optimizeCoeff = optimizeCoeff;
	return *this;
}

SlidingConv& SlidingConv::set_algorithm(SlidingAlgorithm algorithm)
{
	this->content.algorithm = algorithm;
	return *this;
}

SlidingConv& SlidingConv::set_post_process_func(Halide::Func post_process)
{
	this->content.post_process = post_process;
	this->content.has_post_process = true;
	return *this;
}

SlidingConv& SlidingConv::set_target_feature(Halide::Target::Feature feature)
{
	this->content.target.set_feature(feature);
	return *this;
}