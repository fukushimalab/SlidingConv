#include "SlidingConv.h"

using namespace std;
using namespace Halide;
using namespace Halide::Internal;

SlidingConvRefExpr::SlidingConvRefExpr(SlidingConv sc, vector<Expr> a) :sc(sc), args(a) {}

SlidingConvRefExpr::operator Expr() {
	return this->operator[](0);
}

Expr SlidingConvRefExpr::operator[](int i) {
	Function main_func = sc.as_func().function();
	if (i >= main_func.outputs()) {
		cerr << "Could not find output buffer " << i << " in " << sc.name();
		assert(false);
	}
	return Call::make(main_func, args, i);
}