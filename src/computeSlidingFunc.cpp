#include "SlidingConv.h"

#include <Halide.h>

#include "../recfilter/recfilter.h"
#include "computeCoeffDCT.h"
#include "SearchRadiusDCT.h"

#define MY_PI 3.1415926535897932384626433832795

using namespace Halide;
using namespace std;

class SlidingConvFilterDCT
{
	int w;
	int h;
	int d;
	int dctType;
	int order;
	int radius;
	bool withoutDC;

	Halide::Buffer<float> CkGk;
	std::vector<float> shift;

	bool isCoeffOptimized = false;

	// MARK: Init

	void initParametersDCT1(std::function<double(int, double, int)> kernel, bool coeffOptimize)
	{
		double omega = 2.0 * MY_PI / (2.0 * (double)radius + 0.0);
		double k0 = 0.0;
		double n0 = 0.0;

		// Gkの計算
		vector<double> Gk(order + 1);
		if (coeffOptimize) isCoeffOptimized = computeAndOptimizeCoeffDCTFromNormalKernel(dctType, radius, order, kernel, Gk);
		else computeCoeffDCTFromNormalKernel(dctType, radius, order, kernel, Gk);

		CkGk = Buffer<float>((order + 1) * (radius + 1));

		for (int r = 0; r <= radius; ++r)
		{
			for (int k = 0; k <= order; ++k)
			{
				// Ck = cos((k + k0)*omega*(r + n0))
				CkGk((order + 1) * r + k) = cos((k + k0) * omega * (r + n0)) * Gk[k];
			}
		}

		// DCT1で必要なのはCk1とGk
		shift = vector<float>(2 * (order + 1));
		for (int k = 0, i = 0; k <= order; ++k)
		{
			// 2C_{1-n}
			shift[i++] = cos((k + k0) * omega * (1));
			// Gk
			shift[i++] = Gk[k];
		}
	}

	void initParametersDCT3(std::function<double(int, double, int)> kernel, bool coeffOptimize)
	{
		double omega = 2.0 * MY_PI / (2.0 * (double)radius + 2.0);
		double k0 = 0.5;
		double n0 = 0.0;

		// Gkの計算
		vector<double> Gk(order + 1);
		if (coeffOptimize) isCoeffOptimized = computeAndOptimizeCoeffDCTFromNormalKernel(dctType, radius, order, kernel, Gk);
		else computeCoeffDCTFromNormalKernel(dctType, radius, order, kernel, Gk);

		// GkCk
		// 係数をOptimizeしない場合は2R+2でフィルタするため、必要な係数も増える
		if (isCoeffOptimized)
		{
			CkGk = Buffer<float>((order + 1) * (radius + 1));
			for (int r = 0; r <= radius; ++r)
			{
				for (int k = 0; k <= order; ++k)
				{
					// Ck = cos((k + k0)*omega*(r + n0))
					CkGk((order + 1) * r + k) = cos((k + k0) * omega * (r + n0)) * Gk[k];
				}
			}
		}
		else
		{
			CkGk = Buffer<float>((order + 1) * (radius + 2));

			for (int r = 0; r <= radius + 1; ++r)
			{
				for (int k = 0; k <= order; ++k)
				{
					// Ck = cos((k + k0)*omega*(r + n0))
					CkGk((order + 1) * r + k) = cos((k + k0) * omega * (r + n0)) * Gk[k];
				}
			}
		}

		// Zk更新用
		shift = vector<float>(2 * (order + 1));
		for (int k = 0, i = 0; k <= order; ++k)
		{
			// 2C_{1-n0}
			shift[i++] = CkGk((order + 1) * 1 + k) * 2.0 / Gk[k];
			// Gk*Ck
			shift[i++] = CkGk((order + 1) * radius + k);
		}
	}

	void initParametersDCT5(std::function<double(int, double, int)> kernel, bool coeffOptimize)
	{
		double omega = 2.0 * MY_PI / (2.0 * (double)radius + 1.0);
		double k0 = 0.0;
		double n0 = 0.0;

		// Gkの計算
		vector<double> Gk(order + 1);
		if (coeffOptimize) isCoeffOptimized = computeAndOptimizeCoeffDCTFromNormalKernel(dctType, radius, order, kernel, Gk);
		else computeCoeffDCTFromNormalKernel(dctType, radius, order, kernel, Gk);

		CkGk = Buffer<float>((order + 1) * (radius + 1));
		for (int r = 0; r <= radius; ++r)
		{
			for (int k = 0; k <= order; ++k)
			{
				// Ck = cos((k + k0)*omega*(r + n0))
				CkGk((order + 1) * r + k) = cos((k + k0) * omega * (r + n0)) * Gk[k];
			}
		}

		// Zk更新用
		shift = vector<float>(2 * (order + 1));
		for (int k = 0, i = 0; k <= order; ++k)
		{
			// 2C_{1-n}
			shift[i++] = 2.0 * cos((k + k0) * omega * (1));
			// Gk*Ck
			shift[i++] = CkGk((order + 1) * radius + k);
		}
	}

	void initParametersDCT7(std::function<double(int, double, int)> kernel, bool coeffOptimize)
	{
		double omega = 2.0 * MY_PI / (2.0 * (double)radius + 1.0);
		double k0 = 0.5;
		double n0 = 0.0;

		// Gkの計算
		vector<double> Gk(order + 1);
		if (coeffOptimize) isCoeffOptimized = computeAndOptimizeCoeffDCTFromNormalKernel(dctType, radius, order, kernel, Gk);
		else computeCoeffDCTFromNormalKernel(dctType, radius, order, kernel, Gk);

		// GkCk
		CkGk = Buffer<float>((order + 1) * (radius + 1));

		for (int r = 0; r <= radius; ++r)
		{
			for (int k = 0; k <= order; ++k)
			{
				// Ck = cos((k + k0)*omega*r)
				CkGk((order + 1) * r + k) = cos((k + k0) * omega * (r + n0)) * Gk[k];
			}
		}

		// Zk更新用
		shift = vector<float>(2 * (order + 1));
		for (int k = 0, i = 0; k <= order; ++k)
		{
			// 2C_{1-n}
			shift[i++] = CkGk((order + 1) * 1 + k) * 2.0 / Gk[k];
			// Gk*Ck
			shift[i++] = CkGk((order + 1) * radius + k);
		}
	}

	// MARK: Filters

	vector<Func> filterDCT1(Func input, Buffer<float> inputY, int splitX, int splitY, bool has_post_process, Func post_process, ScheduleInfo scheduleX, ScheduleInfo scheduleY, bool asSeparable, Halide::Target target)
	{
		RecFilterDim x("x", w), y("y", h), c("c", d);

		Func input_float_clamped("input_float_clamped");
		input_float_clamped = BoundaryConditions::mirror_interior(input, { { 0, w } , { 0, h} });

		RecFilter slidingRecFilter_Y("SlidingRecFilter_Y", target);
		slidingRecFilter_Y(x, y, c) = 1.f;

		// delta
		vector<Func> deltaFunctions_Y;

		// init
		vector<Func> initFunctions_Y;

		for (int k = 0; k <= order; k++)
		{
			Func delta_Y("deltalll_Y_" + to_string(k));
			// delta = (-1)^k * (f_{x-R-1} + f_{x+R+1} - C^k_1 (f_{x-R} + f_{x+R})
			// (x, y)の時にカーネルに入る画素と出る画素なので、計算基準は一個前の
			delta_Y(x, y, c) = input_float_clamped(x, y - 1 - radius - 1, c) + input_float_clamped(x, y - 1 + radius + 1, c) - shift[2 * k] * (input_float_clamped(x, y - 1 - radius, c) + input_float_clamped(x, y - 1 + radius, c));

			deltaFunctions_Y.push_back(delta_Y);
		}
		slidingRecFilter_Y.set_delta_Func(deltaFunctions_Y);

		// Zk初期値を作成
		for (int k = 0; k <= order; k++)
		{
			Func initial_Y("initial_Y_" + to_string(k));

			Expr condition = y % splitY == 0 || y % splitY == 1;

			// Zk初期値
			Expr trueExpr = input_float_clamped(x, y, c) * CkGk(k);
			for (int i = 1; i <= radius; i++)
			{
				trueExpr += (input_float_clamped(x, y + i, c) + input_float_clamped(x, y - i, c)) * CkGk((order + 1) * i + k);
			}

			initial_Y(x, y, c) = select(condition, trueExpr, undef<float>());

			{
				bool XIIsInnermost = scheduleY.reorder[0] == XI;
				initial_Y
					.compute_at(slidingRecFilter_Y.as_func(), XIIsInnermost ? Var("xo") : Var("xi"))
					.vectorize(Var("x"))
					;
			}

			initFunctions_Y.push_back(initial_Y);
		}

		{
			for (int k = 1; k < initFunctions_Y.size(); k++)
			{
				initFunctions_Y[k - 1].compute_with(initFunctions_Y[k], Var("x"));
			}
		}

		//// 画像がZkになるような係数を作成
		//// Zk(x + 1) = 2C_{1-n}     * Zk(x)        + Gk*Ck * delta - Zk(x - 1)
		//// Zk(x) = 2C_{1-n}     * Zk(x - 1)        + Gk*Ck *delta - Zk(x - 2)
		for (int k = 0; k <= order; k++)
		{
			vector<Expr> coeffZk_Y;

			// feed foward
			// Gk* delta (-1)^kの部分で符号が異なる
			if (k % 2 == 0)
			{
				coeffZk_Y.push_back((Expr)(shift[2 * k + 1]));
			}
			else
			{
				coeffZk_Y.push_back((Expr)(-1 * shift[2 * k + 1]));
			}

			// feed back
			// 2C_{1-n}     * Zk(x)
			coeffZk_Y.push_back((Expr)(2 * shift[2 * k]));
			// - Zk(x - 1)
			coeffZk_Y.push_back((Expr)(-1));

			slidingRecFilter_Y.add_filter(+y, coeffZk_Y, 1);
		}

		slidingRecFilter_Y.set_custom_init_Func(initFunctions_Y);
		slidingRecFilter_Y.algorithm(RecFilter::SlidingSingleMultiDelta);
		slidingRecFilter_Y.split(y, splitY);

		// Schedule

		slidingRecFilter_Y.set_schedule(scheduleY);


		Func out_y = slidingRecFilter_Y.as_func();
		//out_y.print_loop_nest();

		// horizontal

		Func yrealize_clamped("yrealize_clamped");
		if (asSeparable)
		{
			yrealize_clamped = BoundaryConditions::mirror_interior(inputY);
		}
		else
		{
			out_y.compute_root();
			yrealize_clamped = BoundaryConditions::mirror_interior(out_y, { { 0, w } , { 0, h} });
		}

		RecFilter slidingRecFilter_X("SlidingRecFilter_X", target);
		slidingRecFilter_X(x, y, c) = 1.f;

		// delta
		vector<Func> deltaFunctions_X;
		// init
		vector<Func> initFunctions_X;

		for (int k = 0; k <= order; k++)
		{
			Func delta_X("delta_X_" + to_string(k));
			// delta = (-1)^k * (f_{x-R-1} + f_{x+R+1} - C^k_1 (f_{x-R} + f_{x+R})
			// (x, y)の時にカーネルに入る画素と出る画素なので、計算基準は一個前の
			delta_X(x, y, c) = yrealize_clamped(x - 1 - radius - 1, y, c) + yrealize_clamped(x - 1 + radius + 1, y, c) - shift[2 * k] * (yrealize_clamped(x - 1 - radius, y, c) + yrealize_clamped(x - 1 + radius, y, c));

			deltaFunctions_X.push_back(delta_X);
		}
		slidingRecFilter_X.set_delta_Func(deltaFunctions_X);

		for (int k = 0; k <= order; k++)
		{
			Func initial_X("initial_X_" + to_string(k));

			Expr condition = x % splitX == 0 || x % splitX == 1;

			// Zk初期値
			Expr trueExpr = yrealize_clamped(x, y, c) * CkGk(k);
			for (int i = 1; i <= radius; i++)
			{
				trueExpr += (yrealize_clamped(x + i, y, c) + yrealize_clamped(x - i, y, c)) * CkGk((order + 1) * i + k);
			}

			initial_X(x, y, c) = select(condition, trueExpr, undef<float>());

			{
				bool YIIsInnnermost = scheduleX.reorder[0] == YI;
				initial_X
					.compute_at(slidingRecFilter_X.as_func(), YIIsInnnermost ? Var("xo") : Var("yi"))
					.vectorize(Var("x"))
					;
			}

			initFunctions_X.push_back(initial_X);
		}

		{
			for (int k = 1; k < initFunctions_X.size(); k++)
			{
				initFunctions_X[k - 1].compute_with(initFunctions_X[k], Var("x"));
			}
		}

		// 画像がZkになるような係数を作成
		// Zk(x + 1) = 2C_{1-n}     * Zk(x)        + Gk*Ck * delta - Zk(x - 1)
		// Zk(x) = 2C_{1-n}     * Zk(x - 1)        + Gk*Ck * delta - Zk(x - 2)
		for (int k = 0; k <= order; k++)
		{
			vector<Expr> coeffZk_X;

			// feed foward
			// Gk* delta (-1)^kの部分で符号が異なる
			if (k % 2 == 0)
			{
				coeffZk_X.push_back((Expr)(shift[2 * k + 1]));
			}
			else
			{
				coeffZk_X.push_back((Expr)(-1 * shift[2 * k + 1]));
			}

			// feed back
			// 2C_{1-n}     * Zk(x)
			coeffZk_X.push_back((Expr)(2 * shift[2 * k]));
			// - Zk(x - 1)
			coeffZk_X.push_back((Expr)(-1));

			slidingRecFilter_X.add_filter(+x, coeffZk_X, 1);
		}

		slidingRecFilter_X.set_custom_init_Func(initFunctions_X);

		if (has_post_process) {
			slidingRecFilter_X.set_post_process_Func(post_process);
		}
		slidingRecFilter_X.algorithm(RecFilter::SlidingSingleMultiDeltaPostProcess);
		slidingRecFilter_X.split(x, splitX);
		slidingRecFilter_X.set_schedule(scheduleX);


		Func out_x = slidingRecFilter_X.as_func();
		//out_x.print_loop_nest();

		return { out_y, out_x };
	}

	vector<Func> filterDCT1ReduceF0(Func input, Buffer<float> inputY, int splitX, int splitY, bool has_post_process, Func post_process, ScheduleInfo scheduleX, ScheduleInfo scheduleY, bool asSeparable, Halide::Target target)
	{
		RecFilterDim x("x", w), y("y", h), c("c", d);

		Func input_float_clamped("input_float_clamped");
		input_float_clamped = BoundaryConditions::mirror_interior(input, { { 0, w } , { 0, h} });

		RecFilter slidingRecFilter_Y("SlidingRecFilter_Y", target);
		slidingRecFilter_Y(x, y, c) = 1.f;

		// delta
		vector<Func> deltaFunctions_Y;

		// init
		vector<Func> initFunctions_Y;

		Func initial_F0_Y("initial_F0_Y");
		Expr condition = y % splitY == 0;

		// Zk初期値
		Expr trueExpr = input_float_clamped(x, y, c);
		for (int i = 1; i <= radius; i++)
		{
			trueExpr += (input_float_clamped(x, y + i, c) + input_float_clamped(x, y - i, c));
		}

		trueExpr *= CkGk(0);
		initial_F0_Y(x, y, c) = select(condition, trueExpr, undef<float>());
		{
			bool XIIsInnermost = scheduleY.reorder[0] == XI;
			initial_F0_Y
				.compute_at(slidingRecFilter_Y.as_func(), XIIsInnermost ? Var("xo") : Var("xi"))
				.vectorize(Var("x"))
				;
		}
		initFunctions_Y.push_back(initial_F0_Y);

		// F0を別で計算
		Func deltaF0("deltaF0");
		deltaF0(x, y, c) = CkGk(0) * (input_float_clamped(x, y - 1 + radius + 1, c) - input_float_clamped(x, y - 1 - radius, c));
		deltaFunctions_Y.push_back(deltaF0);

		vector<Expr> coeffsF0_Y = { 1, 1 };
		slidingRecFilter_Y.add_filter(+y, coeffsF0_Y, 1);

		for (int k = 1; k <= order; k++)
		{
			Func delta_Y("deltalll_Y_" + to_string(k));
			// delta = (-1)^k * (f_{x-R-1} + f_{x+R+1} - C^k_1 (f_{x-R} + f_{x+R})
			// (x, y)の時にカーネルに入る画素と出る画素なので、計算基準は一個前の
			delta_Y(x, y, c) = input_float_clamped(x, y - 1 - radius - 1, c) + input_float_clamped(x, y - 1 + radius + 1, c) - shift[2 * k] * (input_float_clamped(x, y - 1 - radius, c) + input_float_clamped(x, y - 1 + radius, c));

			deltaFunctions_Y.push_back(delta_Y);
		}
		slidingRecFilter_Y.set_delta_Func(deltaFunctions_Y);

		// Zk初期値を作成
		for (int k = 1; k <= order; k++)
		{
			Func initial_Y("initial_Y_" + to_string(k));

			Expr condition = y % splitY == 0 || y % splitY == 1;

			// Zk初期値
			Expr trueExpr = input_float_clamped(x, y, c) * CkGk(k);
			for (int i = 1; i <= radius; i++)
			{
				trueExpr += (input_float_clamped(x, y + i, c) + input_float_clamped(x, y - i, c)) * CkGk((order + 1) * i + k);
			}

			initial_Y(x, y, c) = select(condition, trueExpr, undef<float>());

			{
				bool XIIsInnermost = scheduleY.reorder[0] == XI;
				initial_Y
					.compute_at(slidingRecFilter_Y.as_func(), XIIsInnermost ? Var("xo") : Var("xi"))
					.vectorize(Var("x"))
					;
			}

			initFunctions_Y.push_back(initial_Y);
		}

		{
			for (int k = 1; k < initFunctions_Y.size(); k++)
			{
				initFunctions_Y[k - 1].compute_with(initFunctions_Y[k], Var("x"));
			}
		}

		//// 画像がZkになるような係数を作成
		//// Zk(x + 1) = 2C_{1-n}     * Zk(x)        + Gk*Ck * delta - Zk(x - 1)
		//// Zk(x) = 2C_{1-n}     * Zk(x - 1)        + Gk*Ck *delta - Zk(x - 2)
		for (int k = 1; k <= order; k++)
		{
			vector<Expr> coeffZk_Y;

			// feed foward
			// Gk* delta (-1)^kの部分で符号が異なる
			if (k % 2 == 0)
			{
				coeffZk_Y.push_back((Expr)(shift[2 * k + 1]));
			}
			else
			{
				coeffZk_Y.push_back((Expr)(-1 * shift[2 * k + 1]));
			}

			// feed back
			// 2C_{1-n}     * Zk(x)
			coeffZk_Y.push_back((Expr)(2 * shift[2 * k]));
			// - Zk(x - 1)
			coeffZk_Y.push_back((Expr)(-1));

			slidingRecFilter_Y.add_filter(+y, coeffZk_Y, 1);
		}

		slidingRecFilter_Y.set_custom_init_Func(initFunctions_Y);
		slidingRecFilter_Y.algorithm(RecFilter::SlidingSingleMultiDelta);
		slidingRecFilter_Y.split(y, splitY);

		// Schedule

		slidingRecFilter_Y.set_schedule(scheduleY);
		

		Func out_y = slidingRecFilter_Y.as_func();
		//out_y.print_loop_nest();

		// horizontal

		Func yrealize_clamped("yrealize_clamped");
		if (asSeparable)
		{
			yrealize_clamped = BoundaryConditions::mirror_interior(inputY);
		}
		else
		{
			out_y.compute_root();
			yrealize_clamped = BoundaryConditions::mirror_interior(out_y, { { 0, w } , { 0, h} });
		}

		RecFilter slidingRecFilter_X("SlidingRecFilter_X", target);
		slidingRecFilter_X(x, y, c) = 1.f;

		// delta
		vector<Func> deltaFunctions_X;
		// init
		vector<Func> initFunctions_X;

		// F0を別で計算
		{
			Func initial_F0_X("initial_F0_X");
			Expr condition = x % splitX == 0;

			// Zk初期値
			Expr trueExpr = yrealize_clamped(x, y, c);
			for (int i = 1; i <= radius; i++)
			{
				trueExpr += (yrealize_clamped(x + i, y, c) + yrealize_clamped(x - i, y, c));
			}

			trueExpr *= CkGk(0);

			initial_F0_X(x, y, c) = select(condition, trueExpr, undef<float>());

			{
				bool YIIsInnnermost = scheduleX.reorder[0] == YI;
				initial_F0_X
					.compute_at(slidingRecFilter_X.as_func(), YIIsInnnermost ? Var("xo") : Var("yi"))
					.vectorize(Var("x"))
					;
			}

			initFunctions_X.push_back(initial_F0_X);

			// F0を別で計算
			Func deltaF0("deltaF0");
			deltaF0(x, y, c) = CkGk(0) * (yrealize_clamped(x - 1 + radius + 1, y, c) - yrealize_clamped(x - 1 - radius, y, c));
			deltaFunctions_X.push_back(deltaF0);

			vector<Expr> coeffsF0_X = { 1, 1 };
			slidingRecFilter_X.add_filter(+x, coeffsF0_X, 1);
		}

		for (int k = 1; k <= order; k++)
		{
			Func delta_X("delta_X_" + to_string(k));
			// delta = (-1)^k * (f_{x-R-1} + f_{x+R+1} - C^k_1 (f_{x-R} + f_{x+R})
			// (x, y)の時にカーネルに入る画素と出る画素なので、計算基準は一個前の
			delta_X(x, y, c) = yrealize_clamped(x - 1 - radius - 1, y, c) + yrealize_clamped(x - 1 + radius + 1, y, c) - shift[2 * k] * (yrealize_clamped(x - 1 - radius, y, c) + yrealize_clamped(x - 1 + radius, y, c));

			deltaFunctions_X.push_back(delta_X);
		}
		slidingRecFilter_X.set_delta_Func(deltaFunctions_X);

		for (int k = 1; k <= order; k++)
		{
			Func initial_X("initial_X_" + to_string(k));

			Expr condition = x % splitX == 0 || x % splitX == 1;

			// Zk初期値
			Expr trueExpr = yrealize_clamped(x, y, c) * CkGk(k);
			for (int i = 1; i <= radius; i++)
			{
				trueExpr += (yrealize_clamped(x + i, y, c) + yrealize_clamped(x - i, y, c)) * CkGk((order + 1) * i + k);
			}

			initial_X(x, y, c) = select(condition, trueExpr, undef<float>());

			{
				bool YIIsInnnermost = scheduleX.reorder[0] == YI;
				initial_X
					.compute_at(slidingRecFilter_X.as_func(), YIIsInnnermost ? Var("xo") : Var("yi"))
					.vectorize(Var("x"))
					;
			}

			initFunctions_X.push_back(initial_X);
		}

		{
			for (int k = 1; k < initFunctions_X.size(); k++)
			{
				initFunctions_X[k - 1].compute_with(initFunctions_X[k], Var("x"));
			}
		}

		// 画像がZkになるような係数を作成
		// Zk(x + 1) = 2C_{1-n}     * Zk(x)        + Gk*Ck * delta - Zk(x - 1)
		// Zk(x) = 2C_{1-n}     * Zk(x - 1)        + Gk*Ck * delta - Zk(x - 2)
		for (int k = 1; k <= order; k++)
		{
			vector<Expr> coeffZk_X;

			// feed foward
			// Gk* delta (-1)^kの部分で符号が異なる
			if (k % 2 == 0)
			{
				coeffZk_X.push_back((Expr)(shift[2 * k + 1]));
			}
			else
			{
				coeffZk_X.push_back((Expr)(-1 * shift[2 * k + 1]));
			}

			// feed back
			// 2C_{1-n}     * Zk(x)
			coeffZk_X.push_back((Expr)(2 * shift[2 * k]));
			// - Zk(x - 1)
			coeffZk_X.push_back((Expr)(-1));

			slidingRecFilter_X.add_filter(+x, coeffZk_X, 1);
		}

		slidingRecFilter_X.set_custom_init_Func(initFunctions_X);

		if (has_post_process) {
			slidingRecFilter_X.set_post_process_Func(post_process);
		}
		slidingRecFilter_X.algorithm(RecFilter::SlidingSingleMultiDeltaPostProcess);
		slidingRecFilter_X.split(x, splitX);
		slidingRecFilter_X.set_schedule(scheduleX);
		

		Func out_x = slidingRecFilter_X.as_func();
		//out_x.print_loop_nest();

		return { out_y, out_x };
	}

	vector<Func> filterDCT1withT(Func input, Buffer<float> inputY, int splitX, int splitY, bool has_post_process, Func post_process, ScheduleInfo scheduleX, ScheduleInfo scheduleY, bool asSeparable, Halide::Target target)
	{
		RecFilterDim x("x", w), y("y", h), c("c", d);

		Func input_float_clamped("input_float_clamped");
		input_float_clamped = BoundaryConditions::mirror_interior(input, { { 0, w } , { 0, h} });

		RecFilter slidingRecFilter_Y("SlidingRecFilter_Y", target);
		slidingRecFilter_Y(x, y, c) = 1.f;

		// delta
		vector<Func> deltaFunctions_Y;
		vector<Func> initFunctions_Y;

		// F0を別で計算
		{
			Func initial_F0_Y("initial_F0_Y");
			Expr condition = y % splitY == 0;

			// Zk初期値
			Expr trueExpr = input_float_clamped(x, y, c) + input_float_clamped(x, y + radius, c);
			for (int i = 1; i <= radius - 1; i++)
			{
				trueExpr += (input_float_clamped(x, y + i, c) + input_float_clamped(x, y - i, c));
			}

			trueExpr *= CkGk(0);

			initial_F0_Y(x, y, c) = select(condition, trueExpr, undef<float>());
			initFunctions_Y.push_back(initial_F0_Y);

			Func deltaF0("deltaF0");
			deltaF0(x, y, c) = CkGk(0) * (input_float_clamped(x, y - 1 + radius + 1, c) - input_float_clamped(x, y - 1 - radius + 1, c));
			deltaFunctions_Y.push_back(deltaF0);

			vector<Expr> coeffsF0_Y = { 1, 1 };
			slidingRecFilter_Y.add_filter(+y, coeffsF0_Y, 1);
		}

		for (int k = 1; k <= order; k++)
		{
			Func delta_Y("delta_Y_" + to_string(k));
			// delta = (-1)^k * (f_{x-R-1} + f_{x+R+1} - C^k_1 (f_{x-R} + f_{x+R})
			// (x, y)の時にカーネルに入る画素と出る画素なので、計算基準は一個前の
			delta_Y(x, y, c) = input_float_clamped(x, y - 1 + radius + 1, c) - input_float_clamped(x, y - 1 - radius + 1, c) + shift[2 * k] * (input_float_clamped(x, y - 1 - radius, c) - input_float_clamped(x, y - 1 + radius, c));

			deltaFunctions_Y.push_back(delta_Y);
		}

		// Zk初期値を作成
		for (int k = 1; k <= order; k++)
		{
			Func initial_Y("initial_Y_" + to_string(k));

			Expr condition = y % splitY == 0 || y % splitY == 1;

			// Zk初期値
			Expr trueExpr = input_float_clamped(x, y, c) * CkGk(k) + input_float_clamped(x, y + radius, c) * CkGk((order + 1) * radius + k);;
			for (int i = 1; i <= radius - 1; i++)
			{
				trueExpr += (input_float_clamped(x, y + i, c) + input_float_clamped(x, y - i, c)) * CkGk((order + 1) * i + k);
			}

			initial_Y(x, y, c) = select(condition, trueExpr, undef<float>());
			{
				bool XIIsInnermost = scheduleY.reorder[0] == XI;
				initial_Y
					.compute_at(slidingRecFilter_Y.as_func(), XIIsInnermost ? Var("xo") : Var("xi"))
					.vectorize(Var("x"))
					;
			}

			initFunctions_Y.push_back(initial_Y);
		}

		{
			for (int k = 1; k < initFunctions_Y.size(); k++)
			{
				initFunctions_Y[k - 1].compute_with(initFunctions_Y[k], Var("x"));
			}
		}

		//// 画像がZkになるような係数を作成
		//// Zk(x + 1) = 2C_{1-n}     * Zk(x)        + Gk*Ck * delta - Zk(x - 1)
		//// Zk(x) = 2C_{1-n}     * Zk(x - 1)        + Gk*Ck *delta - Zk(x - 2)
		for (int k = 1; k <= order; k++)
		{
			vector<Expr> coeffZk_Y;

			// feed foward
			// Gk* delta (-1)^kの部分で符号が異なる
			if (k % 2 == 0)
			{
				coeffZk_Y.push_back((Expr)(shift[2 * k + 1]));
			}
			else
			{
				coeffZk_Y.push_back((Expr)(-1 * shift[2 * k + 1]));
			}

			// feed back
			// 2C_{1-n}     * Zk(x)
			coeffZk_Y.push_back((Expr)(2 * shift[2 * k]));
			// - Zk(x - 1)
			coeffZk_Y.push_back((Expr)(-1));

			slidingRecFilter_Y.add_filter(+y, coeffZk_Y, 1);
		}

		slidingRecFilter_Y.set_delta_Func(deltaFunctions_Y);
		slidingRecFilter_Y.set_custom_init_Func(initFunctions_Y);
		slidingRecFilter_Y.algorithm(RecFilter::SlidingSingleMultiDelta);
		slidingRecFilter_Y.split(y, splitY);

		// Schedule

		slidingRecFilter_Y.set_schedule(scheduleY);
		

		Func out_y = slidingRecFilter_Y.as_func();
		//out_y.print_loop_nest();

		// horizontal

		Func yrealize_clamped("yrealize_clamped");
		if (asSeparable)
		{
			yrealize_clamped = BoundaryConditions::mirror_interior(inputY);
		}
		else
		{
			out_y.compute_root();
			yrealize_clamped = BoundaryConditions::mirror_interior(out_y, { { 0, w } , { 0, h} });
		}

		RecFilter slidingRecFilter_X("SlidingRecFilter_X", target);
		slidingRecFilter_X(x, y, c) = 1.f;

		// delta
		vector<Func> deltaFunctions_X;
		vector<Func> initFunctions_X;

		// F0を別で計算
		{
			Func initial_F0_X("initial_F0_X");
			Expr condition = x % splitX == 0;

			// Zk初期値
			Expr trueExpr = yrealize_clamped(x, y, c) + yrealize_clamped(x + radius, y, c);
			for (int i = 1; i <= radius - 1; i++)
			{
				trueExpr += (yrealize_clamped(x + i, y, c) + yrealize_clamped(x - i, y, c));
			}

			trueExpr *= CkGk(0);

			initial_F0_X(x, y, c) = select(condition, trueExpr, undef<float>());

			{
				bool YIIsInnnermost = scheduleX.reorder[0] == YI;
				initial_F0_X
					.compute_at(slidingRecFilter_X.as_func(), YIIsInnnermost ? Var("xo") : Var("yi"))
					.vectorize(Var("x"))
					;
			}
			initFunctions_X.push_back(initial_F0_X);

			// F0を別で計算
			Func deltaF0("deltaF0");
			deltaF0(x, y, c) = CkGk(0) * (yrealize_clamped(x - 1 + radius + 1, y, c) - yrealize_clamped(x - 1 - radius + 1, y, c));
			deltaFunctions_X.push_back(deltaF0);

			vector<Expr> coeffsF0_X = { 1, 1 };
			slidingRecFilter_X.add_filter(+x, coeffsF0_X, 1);
		}

		for (int k = 1; k <= order; k++)
		{
			Func delta_X("delta_X_" + to_string(k));
			// delta = (-1)^k * (f_{x-R-1} + f_{x+R+1} - C^k_1 (f_{x-R} + f_{x+R})
			// (x, y)の時にカーネルに入る画素と出る画素なので、計算基準は一個前の
			delta_X(x, y, c) = yrealize_clamped(x - 1 + radius + 1, y, c) - yrealize_clamped(x - 1 - radius + 1, y, c) + shift[2 * k] * (yrealize_clamped(x - 1 - radius, y, c) - yrealize_clamped(x - 1 + radius, y, c));

			deltaFunctions_X.push_back(delta_X);
		}

		for (int k = 1; k <= order; k++)
		{
			Func initial_X("initial_X_" + to_string(k));

			Expr condition = x % splitX == 0 || x % splitX == 1;

			// Zk初期値
			Expr trueExpr = yrealize_clamped(x, y, c) * CkGk(k) + yrealize_clamped(x + radius, y, c) * CkGk((order + 1) * radius + k);
			for (int i = 1; i <= radius - 1; i++)
			{
				trueExpr += (yrealize_clamped(x + i, y, c) + yrealize_clamped(x - i, y, c)) * CkGk((order + 1) * i + k);
			}

			initial_X(x, y, c) = select(condition, trueExpr, undef<float>());

			{
				bool YIIsInnnermost = scheduleX.reorder[0] == YI;
				initial_X
					.compute_at(slidingRecFilter_X.as_func(), YIIsInnnermost ? Var("xo") : Var("yi"))
					.vectorize(Var("x"))
					;
			}

			initFunctions_X.push_back(initial_X);
		}

		{
			for (int k = 1; k < initFunctions_X.size(); k++)
			{
				initFunctions_X[k - 1].compute_with(initFunctions_X[k], Var("x"));
			}
		}

		// 画像がZkになるような係数を作成
		// Zk(x + 1) = 2C_{1-n}     * Zk(x)        + Gk*Ck * delta - Zk(x - 1)
		// Zk(x) = 2C_{1-n}     * Zk(x - 1)        + Gk*Ck * delta - Zk(x - 2)
		for (int k = 1; k <= order; k++)
		{
			vector<Expr> coeffZk_X;

			// feed foward
			// Gk* delta (-1)^kの部分で符号が異なる
			if (k % 2 == 0)
			{
				coeffZk_X.push_back((Expr)(shift[2 * k + 1]));
			}
			else
			{
				coeffZk_X.push_back((Expr)(-1 * shift[2 * k + 1]));
			}

			// feed back
			// 2C_{1-n}     * Zk(x)
			coeffZk_X.push_back((Expr)(2 * shift[2 * k]));
			// - Zk(x - 1)
			coeffZk_X.push_back((Expr)(-1));

			slidingRecFilter_X.add_filter(+x, coeffZk_X, 1);
		}

		slidingRecFilter_X.set_delta_Func(deltaFunctions_X);
		slidingRecFilter_X.set_custom_init_Func(initFunctions_X);

		if (has_post_process) {
			slidingRecFilter_X.set_post_process_Func(post_process);
		}
		slidingRecFilter_X.algorithm(RecFilter::SlidingSingleMultiDeltaPostProcess);
		slidingRecFilter_X.split(x, splitX);
		slidingRecFilter_X.set_schedule(scheduleX);
		

		Func out_x = slidingRecFilter_X.as_func();
		//out_x.print_loop_nest();

		return { out_y, out_x };
	}

	vector<Func> filterDCT3(Func input, Buffer<float> inputY, int splitX, int splitY, bool has_post_process, Func post_process, ScheduleInfo scheduleX, ScheduleInfo scheduleY, bool asSeparable, Halide::Target target)
	{
		RecFilterDim x("x", w), y("y", h), c("c", d);

		Func input_float_clamped("input_float_clamped");
		input_float_clamped = BoundaryConditions::mirror_interior(input, { { 0, w } , { 0, h} });

		RecFilter slidingRecFilter_Y("SlidingRecFilter_Y", target);

		// delta
		// delta = f_{x - R - 1} + f_{x + R + 1}
		// (x, y)の時にカーネルに入る画素と出る画素なので、計算基準は一個前の
		slidingRecFilter_Y(x, y, c) = input_float_clamped(x, y - 1 - radius - 1, c) + input_float_clamped(x, y - 1 + radius + 1, c);

		// Zk初期値を作成
		vector<Func> initFunctions_Y;
		for (int k = 0; k <= order; k++)
		{
			Func initial_Y("initial_Y_" + to_string(k));

			Expr condition = y % splitY == 0 || y % splitY == 1;

			// Zk初期値
			Expr trueExpr = input_float_clamped(x, y, c) * CkGk(k);
			for (int i = 1; i <= radius; i++)
			{
				trueExpr += (input_float_clamped(x, y + i, c) + input_float_clamped(x, y - i, c)) * CkGk((order + 1) * i + k);
			}

			initial_Y(x, y, c) = select(condition, trueExpr, undef<float>());
			{
				bool XIIsInnermost = scheduleY.reorder[0] == XI;
				initial_Y
					.compute_at(slidingRecFilter_Y.as_func(), XIIsInnermost ? Var("xo") : Var("xi"))
					.vectorize(Var("x"))
					;
			}

			initFunctions_Y.push_back(initial_Y);
		}

		{
			for (int k = 1; k < initFunctions_Y.size(); k++)
			{
				initFunctions_Y[k - 1].compute_with(initFunctions_Y[k], Var("x"));
			}
		}

		//// 画像がZkになるような係数を作成
		//// Zk(x + 1) = 2C_{1-n}     * Zk(x)        + Gk*Ck * delta - Zk(x - 1)
		//// Zk(x) = 2C_{1-n}     * Zk(x - 1)        + Gk*Ck *delta - Zk(x - 2)
		for (int k = 0; k <= order; k++)
		{
			vector<Expr> coeffZk_Y;

			// feed foward
			// Gk* Ck* delta
			coeffZk_Y.push_back((Expr)(shift[2 * k + 1]));

			// feed back
			// 2C_{1-n}     * Zk(x)
			coeffZk_Y.push_back((Expr)(shift[2 * k]));
			// - Zk(x - 1)
			coeffZk_Y.push_back((Expr)(-1));

			slidingRecFilter_Y.add_filter(+y, coeffZk_Y, 1);
		}

		slidingRecFilter_Y.set_custom_init_Func(initFunctions_Y);
		slidingRecFilter_Y.algorithm(RecFilter::SlidingSingle);
		slidingRecFilter_Y.split(y, splitY);

		// Schedule

		slidingRecFilter_Y.set_schedule(scheduleY);
		

		Func out_y = slidingRecFilter_Y.as_func();
		//out_y.print_loop_nest();

		// horizontal

		Func yrealize_clamped("yrealize_clamped");
		if (asSeparable)
		{
			yrealize_clamped = BoundaryConditions::mirror_interior(inputY);
		}
		else
		{
			out_y.compute_root();
			yrealize_clamped = BoundaryConditions::mirror_interior(out_y, { { 0, w } , { 0, h} });
		}

		RecFilter slidingRecFilter_X("SlidingRecFilter_X", target);
		slidingRecFilter_X(x, y, c) = yrealize_clamped(x - 1 - radius - 1, y, c) + yrealize_clamped(x - 1 + radius + 1, y, c);

		vector<Func> initFunctions_X;
		for (int k = 0; k <= order; k++)
		{
			Func initial_X("initial_X_" + to_string(k));

			Expr condition = x % splitX == 0 || x % splitX == 1;

			// Zk初期値
			Expr trueExpr = yrealize_clamped(x, y, c) * CkGk(k);
			for (int i = 1; i <= radius; i++)
			{
				trueExpr += (yrealize_clamped(x + i, y, c) + yrealize_clamped(x - i, y, c)) * CkGk((order + 1) * i + k);
			}

			initial_X(x, y, c) = select(condition, trueExpr, undef<float>());

			{
				bool YIIsInnnermost = scheduleX.reorder[0] == YI;
				initial_X
					.compute_at(slidingRecFilter_X.as_func(), YIIsInnnermost ? Var("xo") : Var("yi"))
					.vectorize(Var("x"))
					;
			}

			initFunctions_X.push_back(initial_X);
		}

		{
			for (int k = 1; k < initFunctions_X.size(); k++)
			{
				initFunctions_X[k - 1].compute_with(initFunctions_X[k], Var("x"));
			}
		}

		// 画像がZkになるような係数を作成
		// Zk(x + 1) = 2C_{1-n}     * Zk(x)        + Gk*Ck * delta - Zk(x - 1)
		// Zk(x) = 2C_{1-n}     * Zk(x - 1)        + Gk*Ck * delta - Zk(x - 2)
		{
			for (int k = 0; k <= order; k++)
			{
				vector<Expr> coeffZk_X;

				// feed foward
				// Gk* Ck* delta
				coeffZk_X.push_back((Expr)(shift[2 * k + 1]));

				// feed back
				// 2C_{1-n}     * Zk(x)
				coeffZk_X.push_back((Expr)(shift[2 * k]));
				// - Zk(x - 1)
				coeffZk_X.push_back((Expr)(-1));

				slidingRecFilter_X.add_filter(+x, coeffZk_X, 1);
			}
		}

		slidingRecFilter_X.set_custom_init_Func(initFunctions_X);

		if (has_post_process) {
			slidingRecFilter_X.set_post_process_Func(post_process);
		}
		slidingRecFilter_X.algorithm(RecFilter::SlidingSinglePostProcess);
		slidingRecFilter_X.split(x, splitX);
		slidingRecFilter_X.set_schedule(scheduleX);
		

		Func out_x = slidingRecFilter_X.as_func();
		//out_x.print_loop_nest();

		return { out_y, out_x };
	}

	// 実はC_{2R+1}=0なのでTでやってもやらなくても同じ
	vector<Func> filterDCT3withT(Func input, Buffer<float> inputY, int splitX, int splitY, bool has_post_process, Func post_process, ScheduleInfo scheduleX, ScheduleInfo scheduleY, bool asSeparable, Halide::Target target)
	{
		RecFilterDim x("x", w), y("y", h), c("c", d);

		Func input_float_clamped("input_float_clamped");
		input_float_clamped = BoundaryConditions::mirror_interior(input, { { 0, w } , { 0, h} });

		RecFilter slidingRecFilter_Y("SlidingRecFilter_Y", target);

		// delta
		// delta = f_{x - R - 1} + f_{x + R + 1}
		// (x, y)の時にカーネルに入る画素と出る画素なので、計算基準は一個前の
		slidingRecFilter_Y(x, y, c) = input_float_clamped(x, y - 1 - radius - 1, c) + input_float_clamped(x, y - 1 + radius + 1, c);

		// Zk
		vector<RecFilter> GkFks_Y = { };

		// Zk初期値を作成
		vector<Func> initFunctions_Y;
		for (int k = 0; k <= order; k++)
		{
			Func initial_Y("initial_Y_" + to_string(k));

			Expr condition = y % splitY == 0 || y % splitY == 1;

			// Zk初期値
			// ここが2R+1フィルタと少し違う
			Expr trueExpr = input_float_clamped(x, y, c) * CkGk(k) + input_float_clamped(x, y + radius + 1, c) * CkGk((order + 1) * (radius + 1) + k);
			for (int i = 1; i <= radius; i++)
			{
				trueExpr += (input_float_clamped(x, y + i, c) + input_float_clamped(x, y - i, c)) * CkGk((order + 1) * i + k);
			}

			initial_Y(x, y, c) = select(condition, trueExpr, undef<float>());
			{
				bool XIIsInnermost = scheduleY.reorder[0] == XI;
				initial_Y
					.compute_at(slidingRecFilter_Y.as_func(), XIIsInnermost ? Var("xo") : Var("xi"))
					.vectorize(Var("x"))
					;
			}

			initFunctions_Y.push_back(initial_Y);
		}

		{
			for (int k = 1; k < initFunctions_Y.size(); k++)
			{
				initFunctions_Y[k - 1].compute_with(initFunctions_Y[k], Var("x"));
			}
		}

		// 画像がZkになるような係数を作成
		// Zk(x + 1) = 2C_{1-n}     * Zk(x)        + Gk*Ck * delta - Zk(x - 1)
		// Zk(x) = 2C_{1-n}     * Zk(x - 1)        + Gk*Ck *delta - Zk(x - 2)
		for (int k = 0; k <= order; k++)
		{
			vector<Expr> coeffZk_Y;

			// feed foward
			// Gk* Ck* delta
			coeffZk_Y.push_back((Expr)(shift[2 * k + 1]));

			// feed back
			// 2C_{1-n}     * Zk(x)
			coeffZk_Y.push_back((Expr)(shift[2 * k]));
			// - Zk(x - 1)
			coeffZk_Y.push_back((Expr)(-1));

			slidingRecFilter_Y.add_filter(+y, coeffZk_Y, 1);
		}

		slidingRecFilter_Y.set_custom_init_Func(initFunctions_Y);
		slidingRecFilter_Y.algorithm(RecFilter::SlidingSingle);
		slidingRecFilter_Y.split(y, splitY);

		// Schedule

		slidingRecFilter_Y.set_schedule(scheduleY);
		

		Func out_y = slidingRecFilter_Y.as_func();

		// horizontal

		Func yrealize_clamped("yrealize_clamped");
		if (asSeparable)
		{
			yrealize_clamped = BoundaryConditions::mirror_interior(inputY);
		}
		else
		{
			out_y.compute_root();
			yrealize_clamped = BoundaryConditions::mirror_interior(out_y, { { 0, w } , { 0, h} });
		}

		RecFilter slidingRecFilter_X("SlidingRecFilter_X", target);
		slidingRecFilter_X(x, y, c) = yrealize_clamped(x - 1 - radius - 1, y, c) + yrealize_clamped(x - 1 + radius + 1, y, c);

		// Zk初期値を生成
		vector<Func> initFunctions_X;
		for (int k = 0; k <= order; k++)
		{
			Func initial_X("initial_X_" + to_string(k));

			Expr condition = x % splitX == 0 || x % splitX == 1;

			// Zk初期値
			Expr trueExpr = yrealize_clamped(x, y, c) * CkGk(k) + yrealize_clamped(x + radius + 1, y, c) * CkGk((order + 1) * (radius + 1) + k);
			for (int i = 1; i <= radius; i++)
			{
				trueExpr += (yrealize_clamped(x + i, y, c) + yrealize_clamped(x - i, y, c)) * CkGk((order + 1) * i + k);
			}

			initial_X(x, y, c) = select(condition, trueExpr, undef<float>());

			{
				bool YIIsInnnermost = scheduleX.reorder[0] == YI;
				initial_X
					.compute_at(slidingRecFilter_X.as_func(), YIIsInnnermost ? Var("xo") : Var("yi"))
					.vectorize(Var("x"))
					;
			}

			initFunctions_X.push_back(initial_X);
		}

		{
			for (int k = 1; k < initFunctions_X.size(); k++)
			{
				initFunctions_X[k - 1].compute_with(initFunctions_X[k], Var("x"));
			}
		}

		// 画像がZkになるような係数を作成
		// Zk(x + 1) = 2C_{1-n}     * Zk(x)        + Gk*Ck * delta - Zk(x - 1)
		// Zk(x) = 2C_{1-n}     * Zk(x - 1)        + Gk*Ck * delta - Zk(x - 2)
		for (int k = 0; k <= order; k++)
		{
			vector<Expr> coeffZk_X;

			// feed foward
			// Gk* Ck* delta
			coeffZk_X.push_back((Expr)(shift[2 * k + 1]));

			// feed back
			// 2C_{1-n}     * Zk(x)
			coeffZk_X.push_back((Expr)(shift[2 * k]));
			// - Zk(x - 1)
			coeffZk_X.push_back((Expr)(-1));

			slidingRecFilter_X.add_filter(+x, coeffZk_X, 1);
		}

		slidingRecFilter_X.set_custom_init_Func(initFunctions_X);

		if (has_post_process) {
			slidingRecFilter_X.set_post_process_Func(post_process);
		}
		slidingRecFilter_X.algorithm(RecFilter::SlidingSinglePostProcess);
		slidingRecFilter_X.split(x, splitX);
		slidingRecFilter_X.set_schedule(scheduleX);
		

		Func out_x = slidingRecFilter_X.as_func();
		//out_x.print_loop_nest();

		return { out_y, out_x };
	}

	vector<Func> filterDCT5(Func input, Buffer<float> inputY, int splitX, int splitY, bool has_post_process, Func post_process, ScheduleInfo scheduleX, ScheduleInfo scheduleY, bool asSeparable, Halide::Target target)
	{
		RecFilterDim x("x", w), y("y", h), c("c", d);

		Func input_float_clamped("input_float_clamped");
		input_float_clamped = BoundaryConditions::mirror_interior(input, { { 0, w } , { 0, h} });

		RecFilter slidingRecFilter_Y("SlidingRecFilter_Y", target);

		slidingRecFilter_Y(x, y, c) = input_float_clamped(x, y - 1 - radius - 1, c) - input_float_clamped(x, y - 1 - radius, c) - input_float_clamped(x, y - 1 + radius, c) + input_float_clamped(x, y - 1 + radius + 1, c);

		// init
		vector<Func> initFunctions_Y;

		// Zk初期値を作成
		for (int k = 0; k <= order; k++)
		{
			Func initial_Y("initial_Y_" + to_string(k));

			Expr condition = y % splitY == 0 || y % splitY == 1;

			// Zk初期値
			Expr trueExpr = input_float_clamped(x, y, c) * CkGk(k);
			for (int i = 1; i <= radius; i++)
			{
				trueExpr += (input_float_clamped(x, y + i, c) + input_float_clamped(x, y - i, c)) * CkGk((order + 1) * i + k);
			}

			initial_Y(x, y, c) = select(condition, trueExpr, undef<float>());

			{
				bool XIIsInnermost = scheduleY.reorder[0] == XI;
				initial_Y
					.compute_at(slidingRecFilter_Y.as_func(), XIIsInnermost ? Var("xo") : Var("xi"))
					.vectorize(Var("x"))
					;
			}
			initFunctions_Y.push_back(initial_Y);
		}

		{
			for (int k = 1; k < initFunctions_Y.size(); k++)
			{
				initFunctions_Y[k - 1].compute_with(initFunctions_Y[k], Var("x"));
			}
		}

		//// 画像がZkになるような係数を作成
		//// Zk(x + 1) = 2C_{1-n}     * Zk(x)        + Gk*Ck * delta - Zk(x - 1)
		//// Zk(x) = 2C_{1-n}     * Zk(x - 1)        + Gk*Ck *delta - Zk(x - 2)
		for (int k = 0; k <= order; k++)
		{
			vector<Expr> coeffZk_Y;

			// feed foward
			// Gk* Ck* delta
			coeffZk_Y.push_back((Expr)(shift[2 * k + 1]));

			// feed back
			// 2C_{1-n}     * Zk(x)
			coeffZk_Y.push_back((Expr)(shift[2 * k]));
			// - Zk(x - 1)
			coeffZk_Y.push_back((Expr)(-1));

			slidingRecFilter_Y.add_filter(+y, coeffZk_Y, 1);
		}
		slidingRecFilter_Y.set_custom_init_Func(initFunctions_Y);
		slidingRecFilter_Y.algorithm(RecFilter::SlidingSingle);
		slidingRecFilter_Y.split(y, splitY);

		// Schedule

		slidingRecFilter_Y.set_schedule(scheduleY);

		Func out_y = slidingRecFilter_Y.as_func();
		//out_y.print_loop_nest();
		//cout << slidingRecFilter_Y.print_hl_code() << endl;

		// horizontal

		Func yrealize_clamped("yrealize_clamped");
		if (asSeparable)
		{
			yrealize_clamped = BoundaryConditions::mirror_interior(inputY);

			//for only debug Xfilter 
			//yrealize_clamped = BoundaryConditions::mirror_interior(input, { { 0, w } , { 0, h} });
		}
		else
		{
			out_y.compute_root();
			yrealize_clamped = BoundaryConditions::mirror_interior(out_y, { { 0, w } , { 0, h} });
		}

		RecFilter slidingRecFilter_X("SlidingRecFilter_X", target);
		slidingRecFilter_X(x, y, c) = yrealize_clamped(x - 1 - radius - 1, y, c) - yrealize_clamped(x - 1 - radius, y, c) - yrealize_clamped(x - 1 + radius, y, c) + yrealize_clamped(x - 1 + radius + 1, y, c);

		// init
		vector<Func> initFunctions_X;

		for (int k = 0; k <= order; k++)
		{
			Func initial_X("initial_X_" + to_string(k));

			Expr condition = x % splitX == 0 || x % splitX == 1;

			// Zk初期値
			Expr trueExpr = yrealize_clamped(x, y, c) * CkGk(k);
			for (int i = 1; i <= radius; i++)
			{
				trueExpr += (yrealize_clamped(x + i, y, c) + yrealize_clamped(x - i, y, c)) * CkGk((order + 1) * i + k);
			}

			initial_X(x, y, c) = select(condition, trueExpr, undef<float>());
			{
				bool YIIsInnnermost = scheduleX.reorder[0] == YI;
				initial_X
					.compute_at(slidingRecFilter_X.as_func(), YIIsInnnermost ? Var("xo") : Var("yi"))
					.vectorize(Var("x"))
					;
			}

			initFunctions_X.push_back(initial_X);
		}

		{
			for (int k = 1; k < initFunctions_X.size(); k++)
			{
				initFunctions_X[k - 1].compute_with(initFunctions_X[k], Var("x"));
			}
		}

		// 画像がZkになるような係数を作成
		// Zk(x + 1) = 2C_{1-n}     * Zk(x)        + Gk*Ck * delta - Zk(x - 1)
		// Zk(x) = 2C_{1-n}     * Zk(x - 1)        + Gk*Ck * delta - Zk(x - 2)
		{
			for (int k = 0; k <= order; k++)
			{
				vector<Expr> coeffZk_X;

				// feed foward
				// Gk* Ck* delta
				coeffZk_X.push_back((Expr)(shift[2 * k + 1]));

				// feed back
				// 2C_{1-n}     * Zk(x)
				coeffZk_X.push_back((Expr)(shift[2 * k]));
				// - Zk(x - 1)
				coeffZk_X.push_back((Expr)(-1));

				slidingRecFilter_X.add_filter(+x, coeffZk_X, 1);
			}
		}

		slidingRecFilter_X.set_custom_init_Func(initFunctions_X);

		if (has_post_process) {
			slidingRecFilter_X.set_post_process_Func(post_process);
		}
		slidingRecFilter_X.algorithm(RecFilter::SlidingSinglePostProcess);
		slidingRecFilter_X.split(x, splitX);

		slidingRecFilter_X.set_schedule(scheduleX);

		Func out_x = slidingRecFilter_X.as_func();
		//out_x.print_loop_nest();
		//cout << slidingRecFilter_X.print_hl_code() << endl;

		return { out_y, out_x };
	}

	vector<Func> filterDCT5ReduceF0(Func input, Buffer<float> inputY, int splitX, int splitY, bool has_post_process, Func post_process, ScheduleInfo scheduleX, ScheduleInfo scheduleY, bool asSeparable, Halide::Target target)
	{
		RecFilterDim x("x", w), y("y", h), c("c", d);

		Func input_float_clamped("input_float_clamped");
		input_float_clamped = BoundaryConditions::mirror_interior(input, { { 0, w } , { 0, h} });

		RecFilter slidingRecFilter_Y("SlidingRecFilter_Y", target);

		slidingRecFilter_Y(x, y, c) = 1.f;

		// delta
		vector<Func> deltaFunctions_Y;

		// init
		vector<Func> initFunctions_Y;
		{
			Func initial_F0_Y("initial_F0_Y");
			Expr condition = y % splitY == 0;

			// Zk初期値
			Expr trueExpr = input_float_clamped(x, y, c);
			for (int i = 1; i <= radius; i++)
			{
				trueExpr += (input_float_clamped(x, y + i, c) + input_float_clamped(x, y - i, c));
			}

			trueExpr *= CkGk(0);

			initial_F0_Y(x, y, c) = select(condition, trueExpr, undef<float>());

			{
				bool XIIsInnermost = scheduleY.reorder[0] == XI;
				initial_F0_Y
					.compute_at(slidingRecFilter_Y.as_func(), XIIsInnermost ? Var("xo") : Var("xi"))
					.vectorize(Var("x"))
					;
			}
			initFunctions_Y.push_back(initial_F0_Y);

			// F0を別で計算
			Func deltaF0("deltaF0");
			deltaF0(x, y, c) = CkGk(0) * (input_float_clamped(x, y - 1 + radius + 1, c) - input_float_clamped(x, y - 1 - radius, c));
			deltaFunctions_Y.push_back(deltaF0);

			vector<Expr> coeffsF0_Y = { 1, 1 };
			slidingRecFilter_Y.add_filter(+y, coeffsF0_Y, 1);
		}

		for (int k = 1; k <= order; k++)
		{
			Func delta_Y("deltalll_Y_" + to_string(k));
			// delta = f_{x-R-1} - f_{x - R} - f_{x + R} + f_{x + R + 1}
			// (x, y)の時にカーネルに入る画素と出る画素なので、計算基準は一個前の
			delta_Y(x, y, c) = input_float_clamped(x, y - 1 - radius - 1, c) - input_float_clamped(x, y - 1 - radius, c) - input_float_clamped(x, y - 1 + radius, c) + input_float_clamped(x, y - 1 + radius + 1, c);

			deltaFunctions_Y.push_back(delta_Y);
		}
		slidingRecFilter_Y.set_delta_Func(deltaFunctions_Y);

		// Zk初期値を作成
		for (int k = 1; k <= order; k++)
		{
			Func initial_Y("initial_Y_" + to_string(k));

			Expr condition = y % splitY == 0 || y % splitY == 1;

			// Zk初期値
			Expr trueExpr = input_float_clamped(x, y, c) * CkGk(k);
			for (int i = 1; i <= radius; i++)
			{
				trueExpr += (input_float_clamped(x, y + i, c) + input_float_clamped(x, y - i, c)) * CkGk((order + 1) * i + k);
			}

			initial_Y(x, y, c) = select(condition, trueExpr, undef<float>());
	
			{
				bool XIIsInnermost = scheduleY.reorder[0] == XI;
				initial_Y
					.compute_at(slidingRecFilter_Y.as_func(), XIIsInnermost ? Var("xo") : Var("xi"))
					.vectorize(Var("x"))
					;
			}
			initFunctions_Y.push_back(initial_Y);
		}

		{
			for (int k = 1; k < initFunctions_Y.size(); k++)
			{
				initFunctions_Y[k - 1].compute_with(initFunctions_Y[k], Var("x"));
			}
		}

		//// 画像がZkになるような係数を作成
		//// Zk(x + 1) = 2C_{1-n}     * Zk(x)        + Gk*Ck * delta - Zk(x - 1)
		//// Zk(x) = 2C_{1-n}     * Zk(x - 1)        + Gk*Ck *delta - Zk(x - 2)
		for (int k = 1; k <= order; k++)
		{
			vector<Expr> coeffZk_Y;

			// feed foward
			// Gk* Ck* delta
			coeffZk_Y.push_back((Expr)(shift[2 * k + 1]));

			// feed back
			// 2C_{1-n}     * Zk(x)
			coeffZk_Y.push_back((Expr)(shift[2 * k]));
			// - Zk(x - 1)
			coeffZk_Y.push_back((Expr)(-1));

			slidingRecFilter_Y.add_filter(+y, coeffZk_Y, 1);
		}
		slidingRecFilter_Y.set_custom_init_Func(initFunctions_Y);
		slidingRecFilter_Y.algorithm(RecFilter::SlidingSingleMultiDelta);
		slidingRecFilter_Y.split(y, splitY);

		// Schedule

		slidingRecFilter_Y.set_schedule(scheduleY);

		Func out_y = slidingRecFilter_Y.as_func();
		//out_y.print_loop_nest();
		//cout << slidingRecFilter_Y.print_hl_code() << endl;

		// horizontal

		Func yrealize_clamped("yrealize_clamped");
		if (asSeparable)
		{
			yrealize_clamped = BoundaryConditions::mirror_interior(inputY);

			//for only debug Xfilter 
			//yrealize_clamped = BoundaryConditions::mirror_interior(input, { { 0, w } , { 0, h} });
		}
		else
		{
			out_y.compute_root();
			yrealize_clamped = BoundaryConditions::mirror_interior(out_y, { { 0, w } , { 0, h} });
		}

		RecFilter slidingRecFilter_X("SlidingRecFilter_X", target);
		slidingRecFilter_X(x, y, c) = 1.f;

		// delta
		vector<Func> deltaFunctions_X;
		// init
		vector<Func> initFunctions_X;

		// F0を別で計算
		{
			Func initial_F0_X("initial_F0_X");
			Expr condition = x % splitX == 0;

			// Zk初期値
			Expr trueExpr = yrealize_clamped(x, y, c);
			for (int i = 1; i <= radius; i++)
			{
				trueExpr += (yrealize_clamped(x + i, y, c) + yrealize_clamped(x - i, y, c));
			}

			trueExpr *= CkGk(0);

			initial_F0_X(x, y, c) = select(condition, trueExpr, undef<float>());
			{
				bool YIIsInnnermost = scheduleX.reorder[0] == YI;
				initial_F0_X
					.compute_at(slidingRecFilter_X.as_func(), YIIsInnnermost ? Var("xo") : Var("yi"))
					.vectorize(Var("x"))
					;
			}
			initFunctions_X.push_back(initial_F0_X);

			// F0を別で計算
			Func deltaF0("deltaF0");
			deltaF0(x, y, c) = CkGk(0) * (yrealize_clamped(x - 1 + radius + 1, y, c) - yrealize_clamped(x - 1 - radius, y, c));
			deltaFunctions_X.push_back(deltaF0);

			vector<Expr> coeffsF0_X = { 1, 1 };
			slidingRecFilter_X.add_filter(+x, coeffsF0_X, 1);
		}

		for (int k = 1; k <= order; k++)
		{
			Func delta_X("delta_X_" + to_string(k));
			// delta = f_{x-R-1} - f_{x - R} - f_{x + R} + f_{x + R + 1}
			// (x, y)の時にカーネルに入る画素と出る画素なので、計算基準は一個前の
			delta_X(x, y, c) = yrealize_clamped(x - 1 - radius - 1, y, c) - yrealize_clamped(x - 1 - radius, y, c) - yrealize_clamped(x - 1 + radius, y, c) + yrealize_clamped(x - 1 + radius + 1, y, c);

			deltaFunctions_X.push_back(delta_X);
		}
		slidingRecFilter_X.set_delta_Func(deltaFunctions_X);

		for (int k = 1; k <= order; k++)
		{
			Func initial_X("initial_X_" + to_string(k));

			Expr condition = x % splitX == 0 || x % splitX == 1;

			// Zk初期値
			Expr trueExpr = yrealize_clamped(x, y, c) * CkGk(k);
			for (int i = 1; i <= radius; i++)
			{
				trueExpr += (yrealize_clamped(x + i, y, c) + yrealize_clamped(x - i, y, c)) * CkGk((order + 1) * i + k);
			}

			initial_X(x, y, c) = select(condition, trueExpr, undef<float>());
			{
				bool YIIsInnnermost = scheduleX.reorder[0] == YI;
				initial_X
					.compute_at(slidingRecFilter_X.as_func(), YIIsInnnermost ? Var("xo") : Var("yi"))
					.vectorize(Var("x"))
					;
			}

			initFunctions_X.push_back(initial_X);
		}

		{
			for (int k = 1; k < initFunctions_X.size(); k++)
			{
				initFunctions_X[k - 1].compute_with(initFunctions_X[k], Var("x"));
			}
		}

		// 画像がZkになるような係数を作成
		// Zk(x + 1) = 2C_{1-n}     * Zk(x)        + Gk*Ck * delta - Zk(x - 1)
		// Zk(x) = 2C_{1-n}     * Zk(x - 1)        + Gk*Ck * delta - Zk(x - 2)
		{
			for (int k = 1; k <= order; k++)
			{
				vector<Expr> coeffZk_X;

				// feed foward
				// Gk* Ck* delta
				coeffZk_X.push_back((Expr)(shift[2 * k + 1]));

				// feed back
				// 2C_{1-n}     * Zk(x)
				coeffZk_X.push_back((Expr)(shift[2 * k]));
				// - Zk(x - 1)
				coeffZk_X.push_back((Expr)(-1));

				slidingRecFilter_X.add_filter(+x, coeffZk_X, 1);
			}
		}

		slidingRecFilter_X.set_custom_init_Func(initFunctions_X);

		if (has_post_process) {
			slidingRecFilter_X.set_post_process_Func(post_process);
		}
		slidingRecFilter_X.algorithm(RecFilter::SlidingSingleMultiDeltaPostProcess);
		slidingRecFilter_X.split(x, splitX);

		slidingRecFilter_X.set_schedule(scheduleX);

		Func out_x = slidingRecFilter_X.as_func();
		//out_x.print_loop_nest();
		//cout << slidingRecFilter_X.print_hl_code() << endl;

		return { out_y, out_x };
	}

	vector<Func> filterDCT7(Func input, Buffer<float> inputY, int splitX, int splitY, bool has_post_process, Func post_process, ScheduleInfo scheduleX, ScheduleInfo scheduleY, bool asSeparable, Halide::Target target)
	{
		RecFilterDim x("x", w), y("y", h), c("c", d);

		Func input_float_clamped("input_float_clamped");
		input_float_clamped = BoundaryConditions::mirror_interior(input, { { 0, w } , { 0, h} });

		RecFilter slidingRecFilter_Y("SlidingRecFilter_Y", target);

		// delta
		// delta = f_{x-R-1} + f_{x - R} + f_{x + R} + f_{x + R + 1}
		// (x, y)の時にカーネルに入る画素と出る画素なので、計算基準は一個前の画素
		slidingRecFilter_Y(x, y, c) = input_float_clamped(x, y - 1 - radius - 1, c) + input_float_clamped(x, y - 1 - radius, c) + input_float_clamped(x, y - 1 + radius, c) + input_float_clamped(x, y - 1 + radius + 1, c);

		// Zk初期値を作成
		vector<Func> initFunctions_Y;
		for (int k = 0; k <= order; k++)
		{
			Func initial_Y("initial_Y_" + to_string(k));

			Expr condition = y % splitY == 0 || y % splitY == 1;

			// Zk初期値
			Expr trueExpr = input_float_clamped(x, y, c) * CkGk(k);
			for (int i = 1; i <= radius; i++)
			{
				trueExpr += (input_float_clamped(x, y + i, c) + input_float_clamped(x, y - i, c)) * CkGk((order + 1) * i + k);
			}

			initial_Y(x, y, c) = select(condition, trueExpr, undef<float>());
			{
				bool XIIsInnermost = scheduleY.reorder[0] == XI;
				initial_Y
					.compute_at(slidingRecFilter_Y.as_func(), XIIsInnermost ? Var("xo") : Var("xi"))
					.vectorize(Var("x"))
					;
			}

			initFunctions_Y.push_back(initial_Y);
		}

		{
			for (int k = 1; k < initFunctions_Y.size(); k++)
			{
				initFunctions_Y[k - 1].compute_with(initFunctions_Y[k], Var("x"));
			}
		}
		//// 画像がZkになるような係数を作成
		//// Zk(x + 1) = 2C_{1-n}     * Zk(x)        + Gk*Ck * delta - Zk(x - 1)
		//// Zk(x) = 2C_{1-n}     * Zk(x - 1)        + Gk*Ck *delta - Zk(x - 2)
		for (int k = 0; k <= order; k++)
		{
			vector<Expr> coeffZk_Y;

			// feed foward
			// Gk* Ck* delta
			coeffZk_Y.push_back((Expr)(shift[2 * k + 1]));

			// feed back
			// 2C_{1-n}     * Zk(x)
			coeffZk_Y.push_back((Expr)(shift[2 * k]));
			// - Zk(x - 1)
			coeffZk_Y.push_back((Expr)(-1));

			slidingRecFilter_Y.add_filter(+y, coeffZk_Y, 1);
		}

		slidingRecFilter_Y.set_custom_init_Func(initFunctions_Y);
		slidingRecFilter_Y.algorithm(RecFilter::SlidingSingle);
		slidingRecFilter_Y.split(y, splitY);

		// Schedule

		slidingRecFilter_Y.set_schedule(scheduleY);
		

		Func out_y = slidingRecFilter_Y.as_func();
		//out_y.print_loop_nest();

		// horizontal

		Func yrealize_clamped("yrealize_clamped");
		if (asSeparable)
		{
			yrealize_clamped = BoundaryConditions::mirror_interior(inputY);
		}
		else
		{
			out_y.compute_root();
			yrealize_clamped = BoundaryConditions::mirror_interior(out_y, { { 0, w } , { 0, h} });
		}

		RecFilter slidingRecFilter_X("SlidingRecFilter_X", target);
		// delta = f_{x-R-1} + f_{x - R} + f_{x + R} + f_{x + R + 1}
		// (x, y)の時にカーネルに入る画素と出る画素なので、計算基準は一個前の画素
		slidingRecFilter_X(x, y, c) = yrealize_clamped(x - 1 - radius - 1, y, c) + yrealize_clamped(x - 1 - radius, y, c) + yrealize_clamped(x - 1 + radius, y, c) + yrealize_clamped(x - 1 + radius + 1, y, c);

		vector<Func> initFunctions_X;
		for (int k = 0; k <= order; k++)
		{
			Func initial_X("initial_X_" + to_string(k));

			Expr condition = x % splitX == 0 || x % splitX == 1;

			// Zk初期値
			Expr trueExpr = yrealize_clamped(x, y, c) * CkGk(k);
			for (int i = 1; i <= radius; i++)
			{
				trueExpr += (yrealize_clamped(x + i, y, c) + yrealize_clamped(x - i, y, c)) * CkGk((order + 1) * i + k);
			}

			initial_X(x, y, c) = select(condition, trueExpr, undef<float>());

			{
				bool YIIsInnnermost = scheduleX.reorder[0] == YI;
				initial_X
					.compute_at(slidingRecFilter_X.as_func(), YIIsInnnermost ? Var("xo") : Var("yi"))
					.vectorize(Var("x"))
					;
			}

			initFunctions_X.push_back(initial_X);
		}

		{
			for (int k = 1; k < initFunctions_X.size(); k++)
			{
				initFunctions_X[k - 1].compute_with(initFunctions_X[k], Var("x"));
			}
		}

		// 画像がZkになるような係数を作成
		// Zk(x + 1) = 2C_{1-n}     * Zk(x)        + Gk*Ck * delta - Zk(x - 1)
		// Zk(x) = 2C_{1-n}     * Zk(x - 1)        + Gk*Ck * delta - Zk(x - 2)
		{
			for (int k = 0; k <= order; k++)
			{
				vector<Expr> coeffZk_X;

				// feed foward
				// Gk* Ck* delta
				coeffZk_X.push_back((Expr)(shift[2 * k + 1]));

				// feed back
				// 2C_{1-n}     * Zk(x)
				coeffZk_X.push_back((Expr)(shift[2 * k]));
				// - Zk(x - 1)
				coeffZk_X.push_back((Expr)(-1));

				slidingRecFilter_X.add_filter(+x, coeffZk_X, 1);
			}
		}

		slidingRecFilter_X.set_custom_init_Func(initFunctions_X);

		if (has_post_process) {
			slidingRecFilter_X.set_post_process_Func(post_process);
		}
		slidingRecFilter_X.algorithm(RecFilter::SlidingSinglePostProcess);
		slidingRecFilter_X.split(x, splitX);
		slidingRecFilter_X.set_schedule(scheduleX);
		

		Func out_x = slidingRecFilter_X.as_func();
		//out_x.print_loop_nest();

		return { out_y, out_x };
	}

	int getOptimizeRadiusDCT(const bool isGoldenSelectionSearch, std::function<double(int, double, int)> kernel, bool coeffOptimize)
	{
		int argmin_r = 0;

		int rmin = 1;
		int rmax = 60;

		SearchRadiusDCT Search(dctType, order, kernel, coeffOptimize);

		if (isGoldenSelectionSearch)
		{
			argmin_r = Search.goldenSearch(rmin, rmax);
		}
		else
		{
			argmin_r = Search.linearSearch(rmin, rmax);
		}

		return argmin_r;
	}

public:

	SlidingConvFilterDCT(int w, int h, int d, int dctType, bool withoutDC, int order, int radius, std::function<double(int, double, int)> kernel, bool coeffOptimize = true) : w(w), h(h), d(d), dctType(dctType), withoutDC(withoutDC), order(order), radius(radius)
	{

		if (radius == 0)
		{
			this->radius = getOptimizeRadiusDCT(false, kernel, coeffOptimize);
			//cout << "radius optimized DCT-" << dctType << " radius:" << this->radius << endl;
		}

		switch (dctType)
		{
		case 1:
			initParametersDCT1(kernel, coeffOptimize);
			break;
		case 3:
			initParametersDCT3(kernel, coeffOptimize);
			break;
		case 5:
			initParametersDCT5(kernel, coeffOptimize);
			break;
		case 7:
			initParametersDCT7(kernel, coeffOptimize);
			break;
		}
	};

	std::vector<Halide::Func> filter(Func input, Halide::Buffer<float> inputY, int splitX, int splitY, bool has_post_process, Func post_process, ScheduleInfo scheduleX, ScheduleInfo scheduleY, bool asSeparable, Halide::Target target)
	{
		switch (dctType)
		{
		case 1:
			if (withoutDC) 
			{
				return filterDCT1(input, inputY, splitX, splitY, has_post_process, post_process, scheduleX, scheduleY, asSeparable, target);
			}
			else
			{
				if (isCoeffOptimized) return filterDCT1ReduceF0(input, inputY, splitX, splitY, has_post_process, post_process, scheduleX, scheduleY, asSeparable, target);
				else return filterDCT1withT(input, inputY, splitX, splitY, has_post_process, post_process, scheduleX, scheduleY, asSeparable, target);
			}
			
		case 3:
			if (isCoeffOptimized) return filterDCT3(input, inputY, splitX, splitY, has_post_process, post_process, scheduleX, scheduleY, asSeparable, target);
			else return filterDCT3withT(input, inputY, splitX, splitY, has_post_process, post_process, scheduleX, scheduleY, asSeparable, target);
		case 5:
			if (withoutDC) 
			{
				return filterDCT5(input, inputY, splitX, splitY, has_post_process, post_process, scheduleX, scheduleY, asSeparable, target);
			}
			else
			{
				return filterDCT5ReduceF0(input, inputY, splitX, splitY, has_post_process, post_process, scheduleX, scheduleY, asSeparable, target);
			}
		case 7:
			return filterDCT7(input, inputY, splitX, splitY, has_post_process, post_process, scheduleX, scheduleY, asSeparable, target);
		}
	}

};

vector<Func> SlidingConv::computeSlidingFunc(Buffer<float> inputX, bool asSeparable)
{
	switch (this->content.algorithm)
	{
	case NONE:
		throw Error("need set slidingAlgorithm");

	case DCT1:
	{
		SlidingConvFilterDCT filter = SlidingConvFilterDCT(content.width, content.height, content.channels, 1, false, content.order, content.radius, content.kernel, content.optimizeCoeff);
		vector<Func> filters = filter.filter(Func(content.internalFunction), inputX, content.scheduleX.splitX, content.scheduleY.splitY, content.has_post_process, content.post_process, content.scheduleX, content.scheduleY, asSeparable, content.target);
		return filters;
	}

	case DCT1WithoutDC:
	{
		SlidingConvFilterDCT filter = SlidingConvFilterDCT(content.width, content.height, content.channels, 1, true, content.order, content.radius, content.kernel, content.optimizeCoeff);
		vector<Func> filters = filter.filter(Func(content.internalFunction), inputX, content.scheduleX.splitX, content.scheduleY.splitY, content.has_post_process, content.post_process, content.scheduleX, content.scheduleY, asSeparable, content.target);
		return filters;
	}

	case DCT3:
	{
		SlidingConvFilterDCT filter = SlidingConvFilterDCT(content.width, content.height, content.channels, 3, false, content.order, content.radius, content.kernel, content.optimizeCoeff);
		vector<Func> filters = filter.filter(Func(content.internalFunction), inputX, content.scheduleX.splitX, content.scheduleY.splitY, content.has_post_process, content.post_process, content.scheduleX, content.scheduleY, asSeparable, content.target);
		return filters;
	}

	case DCT5:
	{
		SlidingConvFilterDCT filter = SlidingConvFilterDCT(content.width, content.height, content.channels, 5, false, content.order, content.radius, content.kernel, content.optimizeCoeff);
		vector<Func> filters = filter.filter(Func(content.internalFunction), inputX, content.scheduleX.splitX, content.scheduleY.splitY, content.has_post_process, content.post_process, content.scheduleX, content.scheduleY, asSeparable, content.target);
		return filters;
	}

	case DCT5WithoutDC:
	{
		SlidingConvFilterDCT filter = SlidingConvFilterDCT(content.width, content.height, content.channels, 5, true, content.order, content.radius, content.kernel, content.optimizeCoeff);
		vector<Func> filters = filter.filter(Func(content.internalFunction), inputX, content.scheduleX.splitX, content.scheduleY.splitY, content.has_post_process, content.post_process, content.scheduleX, content.scheduleY, asSeparable, content.target);
		return filters;
	}

	case DCT7:
	{
		SlidingConvFilterDCT filter = SlidingConvFilterDCT(content.width, content.height, content.channels, 7, false, content.order, content.radius, content.kernel, content.optimizeCoeff);
		vector<Func> filters = filter.filter(Func(content.internalFunction), inputX, content.scheduleX.splitX, content.scheduleY.splitY, content.has_post_process, content.post_process, content.scheduleX, content.scheduleY, asSeparable, content.target);
		return filters;
	}
	}
}