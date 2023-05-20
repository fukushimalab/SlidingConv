#include "SlidingConv.h"

#include <Halide.h>

using namespace std;
using namespace Halide;
using namespace Halide::Internal;

SlidingConv& SlidingConv::operator=(const SlidingConv& sc)
{
	this->content = sc.content;
	return *this;
}

SlidingConvRefVar SlidingConv::operator()(Halide::Var x)
{
	return SlidingConvRefVar(*this, { x });
}

SlidingConvRefVar SlidingConv::operator()(Halide::Var x, Halide::Var y)
{
	return SlidingConvRefVar(*this, { x, y });
}

SlidingConvRefVar SlidingConv::operator()(Halide::Var x, Halide::Var y, Halide::Var z)
{
	return SlidingConvRefVar(*this, { x, y, z });
}

SlidingConvRefVar SlidingConv::operator()(std::vector<Halide::Var> x)
{
	return SlidingConvRefVar(*this, x);
}


SlidingConvRefExpr SlidingConv::operator()(Halide::Expr x)
{
	return SlidingConvRefExpr(*this, { x });
}

SlidingConvRefExpr SlidingConv::operator()(Halide::Expr x, Halide::Expr y)
{
	return SlidingConvRefExpr(*this, { x, y });
}

SlidingConvRefExpr SlidingConv::operator()(Halide::Expr x, Halide::Expr y, Halide::Expr	z)
{
	return SlidingConvRefExpr(*this, { x, y, z });
}

SlidingConvRefExpr SlidingConv::operator()(std::vector<Halide::Expr> x)
{
	return SlidingConvRefExpr(*this, x);
}


void SlidingConv::define(std::vector<Halide::Var> pure_args, std::vector<Halide::Expr> pure_def)
{
	assert(!pure_args.empty());
	assert(!pure_def.empty());

	content.type = pure_def[0].type();
	for (int i = 1; i < pure_def.size(); i++)
	{
		if (content.type != pure_def[i].type())
		{
			cerr << "Type of all Tuple elements in filter definition must be same.\n";
			assert(false);
		}
	}

	if (content.isDefined)
	{
		cerr << "SlidingConv" << content.name << " already defined\n";
		assert(false);
	}

	vector<string> args;
	for (Var a : pure_args)
	{
		args.push_back(a.name());
	}

	Function f(content.name);
	f.define(args, pure_def);

	content.internalFunction = f;
	content.isDefined = true;
}
