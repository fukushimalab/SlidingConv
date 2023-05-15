#include "SlidingConv.h"

using namespace Halide;
using namespace Halide::Internal;
using namespace std;

SlidingConvRefVar::SlidingConvRefVar(SlidingConv sc, vector<Halide::Var> a) :
	sc(sc), args(a) {}

void SlidingConvRefVar::operator=(Expr pure_def) {
	sc.define(args, { pure_def });
}

void SlidingConvRefVar::operator=(vector<Expr> pure_def) {
	sc.define(args, pure_def);
}

void SlidingConvRefVar::operator=(FuncRef pure_def) {
	sc.define(args, { Expr(pure_def) });
}

void SlidingConvRefVar::operator=(SlidingConvRefVar pure_def) {
	Function f = pure_def.sc.as_func().function();
	vector<Expr> values, call_args;
	for (Expr e : pure_def.args) {
		call_args.push_back(e);
	}
	for (int i = 0; i < f.outputs(); i++) {
		values.push_back(Call::make(f, call_args, i));
	}

	sc.define(args, values);
}

SlidingConvRefVar::operator Halide::Expr(void) {
	return this->operator[](0);
}

Expr SlidingConvRefVar::operator[](int i) {
	Function main_func = sc.as_func().function();
	vector<Expr> expr_args;
	for (Var a : args) {
		expr_args.push_back(a);
	}
	if (i >= main_func.outputs()) {
		cerr << "Could not find output buffer " << i << " in " << sc.name();
		assert(false);
	}
	return Call::make(main_func, expr_args, i);
}

