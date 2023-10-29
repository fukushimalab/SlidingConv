#pragma once

#include <Halide.h>
#include "ScheduleInfo.h"

class SlidingConvRefSchedule;
class SlidingConvRefExpr;
class SlidingConvRefVar;

enum SlidingAlgorithm
{
	NONE, // no selection

	DCT1,
	DCT1WithoutDC,
	DCT3,
	DCT5,
	DCT5WithoutDC,
	DCT7,
};

struct SlidingConvContent
{
public:
	std::string name = "";
	int width = 0;
	int height = 0;
	int channels = 0;

	std::function<double(int, double, int)> kernel = NULL;
	int order = 0;
	int radius = 0;
	bool optimizeCoeff = false;

	ScheduleInfo scheduleX;
	ScheduleInfo scheduleY;

	Halide::Func post_process;
	bool has_post_process = false;

	SlidingAlgorithm algorithm = SlidingAlgorithm::NONE;

	// MARK: - Function

	Halide::Internal::Function internalFunction;
	Halide::Type type = Halide::Type();
	Halide::Target target = Halide::get_jit_target_from_environment();
	bool isDefined = false;

	SlidingConvContent(std::string name, int width, int height, int channels): name(name), width(width), height(height), channels(channels) {}
	SlidingConvContent() { }
};

class SlidingConv
{
public:

	// MARK: - Getters

	std::string name();

	// MARK: - Seters

	SlidingConv(std::string name, int width, int height, int channels);

	SlidingConv& set_kernel(std::function<double(int, double, int)> kernel);
	SlidingConv& set_order(int order);
	SlidingConv& set_radius(int radius);
	SlidingConv& set_optimizeCoeff(bool optimizeCoeff);
	SlidingConv& set_algorithm(SlidingAlgorithm algorithm);
	SlidingConv& set_post_process_func(Halide::Func post_process_func);
	SlidingConv& set_target_feature(Halide::Target::Feature feature);

	// MARK: - Schedule

	SlidingConvRefSchedule scheduleX();
	SlidingConvRefSchedule scheduleY();
	void cpu_auto_schedule();
	void gpu_auto_schedule();

	// MARK: - Inptus

	SlidingConv& operator=(const SlidingConv& sc);

	SlidingConvRefVar operator()(Halide::Var x);
	SlidingConvRefVar operator()(Halide::Var x, Halide::Var y);
	SlidingConvRefVar operator()(Halide::Var x, Halide::Var y, Halide::Var z);
	SlidingConvRefVar operator()(std::vector<Halide::Var> x);

	SlidingConvRefExpr operator()(Halide::Expr x);
	SlidingConvRefExpr operator()(Halide::Expr x, Halide::Expr y);
	SlidingConvRefExpr operator()(Halide::Expr x, Halide::Expr y, Halide::Expr	z);
	SlidingConvRefExpr operator()(std::vector<Halide::Expr> x);

	void define(std::vector<Halide::Var> pure_args, std::vector<Halide::Expr> pure_def);

	// MARK: - Outputs

	Halide::Func as_func();
	void prepareForRealize();
	void realize(Halide::Buffer<float> output);

private:

	bool preparedForRealize = false;
	Halide::Buffer<float> inputXForRealize;
	std::vector<Halide::Func> funcsForRealize;

	// 参照じゃないと上手く動かない
	SlidingConvContent& content = *(new SlidingConvContent());

	std::vector<Halide::Func> computeSlidingFunc(Halide::Buffer<float> inputX, bool asSeparable);
};

class SlidingConvRefSchedule
{
private:
	SlidingConvContent& content;
	bool isX;
public:

	SlidingConvRefSchedule(SlidingConvContent& content, bool isX);

	SlidingConvRefSchedule& tile(int splitX, int splitY);
	SlidingConvRefSchedule& tile(int splitXAndY);
	SlidingConvRefSchedule& parallel(ScheduleTag tag);
	SlidingConvRefSchedule& parallel(std::vector<ScheduleTag> tag);
	SlidingConvRefSchedule& reorder(std::vector<ScheduleTag>);
	SlidingConvRefSchedule& vectorize(ScheduleTag tag, int factor);
	SlidingConvRefSchedule& vectorize(ScheduleTag tag);
	SlidingConvRefSchedule& unroll(ScheduleTag tag, int factor);
	SlidingConvRefSchedule& unroll(ScheduleTag tag);

	// for GPU
	SlidingConvRefSchedule& gpuBlocks(std::vector<ScheduleTag>);
	SlidingConvRefSchedule& gpuThreads(std::vector<ScheduleTag>);
};

class SlidingConvRefExpr
{
private:
	SlidingConv sc;
	std::vector<Halide::Expr> args;
public:
	SlidingConvRefExpr(SlidingConv s, std::vector<Halide::Expr> a);
	operator Halide::Expr(void);
	Halide::Expr operator[](int);
};

class SlidingConvRefVar
{
private:
	SlidingConv sc;
	std::vector<Halide::Var> args;
public:
	SlidingConvRefVar(SlidingConv s, std::vector<Halide::Var> a);

	// MARK: - Define

	void operator=(Halide::Expr pure_def);
	void operator=(Halide::FuncRef pure_def);
	void operator=(SlidingConvRefVar pure_def);
	void operator=(std::vector<Halide::Expr> pure_def);

	operator Halide::Expr(void);

	Halide::Expr operator[](int);
};