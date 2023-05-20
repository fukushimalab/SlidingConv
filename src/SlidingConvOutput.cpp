#include "SlidingConv.h"

#include <Halide.h>

using namespace std;
using namespace Halide;
using namespace Halide::Internal;

Func SlidingConv::as_func()
{
	Buffer<float> null;
	return computeSlidingFunc(null, false)[1];
}

void SlidingConv::prepareForRealize()
{
	inputXForRealize = Buffer<float>(content.width, content.height, content.channels);
	funcsForRealize = computeSlidingFunc(inputXForRealize, true);
	preparedForRealize = true;
}

void SlidingConv::realize(Halide::Buffer<float> output)
{
	funcsForRealize[0].realize(inputXForRealize, content.target);
	funcsForRealize[1].realize(output, content.target);
}