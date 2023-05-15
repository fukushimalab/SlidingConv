#include "recfilter.h"
#include "recfilter_internals.h"
#include "modifiers.h"
#include "timing.h"

#define AUTO_SCHEDULE_MAX_DIMENSIONS 3

using std::string;
using std::cerr;
using std::endl;
using std::vector;
using std::map;
using std::ostream;
using std::stringstream;

namespace Halide
{
	namespace Internal
	{
		template<>
		RefCount& ref_count<RecFilterContents>(const RecFilterContents* f)
		{
			return f->ref_count;
		}
		template<>
		void destroy<RecFilterContents>(const RecFilterContents* f)
		{
			delete f;
		}
	}
}

// -----------------------------------------------------------------------------

using namespace Halide;
using namespace Halide::Internal;

// -----------------------------------------------------------------------------

int RecFilter::max_threads_per_cuda_warp = 0;
int RecFilter::vectorization_width = 0;

void RecFilter::set_max_threads_per_cuda_warp(int v)
{
	if (v % 32 == 0)
	{
		max_threads_per_cuda_warp = v;
	}
	else
	{
		cerr << "RecFilter::set_max_threads_per_cuda_warp(): max "
			"threads in CUDA warp must be a multiple of 32" << endl;
		assert(false);
	}
}

void RecFilter::set_vectorization_width(int v)
{
	if (v == 1 || v == 2 || v == 4 || v == 8 || v == 16 || v == 32 || v == 64 || v == 128)
	{
		vectorization_width = v;
	}
	else
	{
		cerr << "RecFilter::set_vectorization_width(): vectorization width "
			<< "must be a power of 2, 4, 8, 16, 32 or 64" << endl;
		assert(false);
	}
}

void RecFilter::switch_order_of_storage(vector<Func> funcs)
{
	RecFilterFunc& rF = internal_function(name());
	vector<string> args = rF.func.args();
	Var x(args[0]), y(args[1]);
	//for (Func consumer_Func : consumer_Funcs) {
	Func f_in = Func(rF.func).in(funcs);
	f_in.compute_at(funcs[0], x).reorder_storage(y, x)
		.unroll(y, 64).parallel(y).vectorize(x, 16);
	return;
	stringstream s;

	//std::cout << str << std::endl;
	rF.pure_schedule.push_back(s.str());
}

void RecFilter::switch_order_of_storage(RecFilter consumer)
{
	Var consumer_var;
	bool flg = false;;

	vector<Func> consumer_Funcs;

	vector<string> consumer_finals = consumer.internal_functions(FINAL);
	for (int i = 0; i < consumer_finals.size(); i++)
	{
		RecFilterFunc consumer_final_rF = consumer.internal_function(consumer_finals[i]); // Fina
		Function consumer_final_F = consumer_final_rF.func;
		consumer_Funcs.push_back(Func(consumer_final_F));
		if (!flg)
		{
			bool found = false;
			map<string, VarTag> pure_var_category = consumer_final_rF.pure_var_category;
			map<string, VarTag>::iterator vit = pure_var_category.begin();
			map<string, VarTag>::iterator vend = pure_var_category.end();
			for (; !found && vit != vend; vit++)
			{
				if (vit->second == VarTag(OUTER, 0))
				{
					found = true;
					consumer_var = Var(vit->first);
				}
				//std::cout << vit->first << " : " << vit->second << std::endl;
			}

			if (!found)
			{
				cerr << name() << " cannot be computed locally" << endl;
				assert(false);
			}
			flg = true;
		}
	}

	vector<string> consumer_inits = consumer.internal_functions(INIT);
	for (int i = 0; i < consumer_inits.size(); i++)
	{
		RecFilterFunc consumer_init_rF = consumer.internal_function(consumer_inits[i]); // Init
		Function consumer_init_F = consumer_init_rF.func;
		consumer_Funcs.push_back(Func(consumer_init_F));
	}

	switch_order_of_storage(consumer_Funcs);
	//}
}

// -----------------------------------------------------------------------------

RecFilterRefVar::RecFilterRefVar(RecFilter r, std::vector<RecFilterDim> a) :
	rf(r), args(a) {}

void RecFilterRefVar::operator=(Expr pure_def)
{
	rf.define(args, { pure_def });
}

void RecFilterRefVar::operator=(const Tuple& pure_def)
{
	rf.define(args, pure_def.as_vector());
}

void RecFilterRefVar::operator=(vector<Expr> pure_def)
{
	rf.define(args, pure_def);
}

void RecFilterRefVar::operator=(FuncRef pure_def)
{
	rf.define(args, { Expr(pure_def) });
}

void RecFilterRefVar::operator=(RecFilterRefVar pure_def)
{
	Function f = pure_def.rf.as_func().function();
	vector<Expr> values;
	vector<Expr> call_args;
	for (Expr e : pure_def.args)
	{
		call_args.push_back(e);
	}
	for (int i = 0; i < f.outputs(); i++)
	{
		values.push_back(Internal::Call::make(f, call_args, i));
	}

	rf.define(args, values);
}

//void RecFilterRefVar::operator=(FuncRefExpr pure_def)
//{
//    rf.define(args, {Expr(pure_def)});
//}

RecFilterRefVar::operator Expr(void)
{
	return this->operator[](0);
}

Expr RecFilterRefVar::operator[](int i)
{
	Function main_func = rf.as_func().function();
	vector<Expr> expr_args;
	for (int j = 0; j < args.size(); j++)
	{
		expr_args.push_back(args[j]);
	}
	if (i >= main_func.outputs())
	{
		cerr << "Could not find output buffer " << i
			<< " in recursive filter " << rf.name();
		assert(false);
	}
	return Call::make(main_func, expr_args, i);
}

RecFilterRefExpr::RecFilterRefExpr(RecFilter r, std::vector<Expr> a) :
	rf(r), args(a) {}

RecFilterRefExpr::operator Expr(void)
{
	return this->operator[](0);
}

Expr RecFilterRefExpr::operator[](int i)
{
	Function main_func = rf.as_func().function();
	if (i >= main_func.outputs())
	{
		cerr << "Could not find output buffer " << i
			<< " in recursive filter " << rf.name();
		assert(false);
	}
	return Call::make(main_func, args, i);
}

// -----------------------------------------------------------------------------
RecFilter::RecFilter(string name)
{
	RecFilter(name, get_jit_target_from_environment());
}

RecFilter::RecFilter(string name, Halide::Target target)
{
	contents = new RecFilterContents;
	if (name.empty())
	{
		contents.get()->name = unique_name("R");
	}
	else
	{
		contents.get()->name = unique_name(name);
	}
	contents.get()->tiled = false;
	contents.get()->finalized = false;
	contents.get()->compiled = false;
	contents.get()->clamped_border = false;
	contents.get()->feedfwd_coeff = Buffer<float>(0, 0);
	contents.get()->feedback_coeff = Buffer<float>(0, 0);
	contents.get()->feedfwd_coeff_expr = { std::vector<Halide::Expr>(0),std::vector<Halide::Expr>(0) };
	contents.get()->feedback_coeff_expr = { std::vector<Halide::Expr>(0),std::vector<Halide::Expr>(0) };

	contents.get()->target = target;
	if (contents.get()->target.to_string().empty())
	{
		cerr << "Warning: HL_JIT_TARGET not set, using default" << endl;
	}

	this->Algorithm = Original;
	this->redundant = -1;
	this->parallel = true;
	this->is_inlined = false;
	this->use_custom_init = false;
}

RecFilter& RecFilter::operator=(const RecFilter& f)
{
	contents = f.contents;
	return *this;
}

string RecFilter::name(void) const
{
	return contents.get()->name;
}

RecFilterRefVar RecFilter::operator()(RecFilterDim x)
{
	return RecFilterRefVar(*this, { x });
}
RecFilterRefVar RecFilter::operator()(RecFilterDim x, RecFilterDim y)
{
	return RecFilterRefVar(*this, { x, y });
}
RecFilterRefVar RecFilter::operator()(RecFilterDim x, RecFilterDim y, RecFilterDim z)
{
	return RecFilterRefVar(*this, { x, y, z });
}
RecFilterRefVar RecFilter::operator()(vector<RecFilterDim> x)
{
	return RecFilterRefVar(*this, x);
}

RecFilterRefExpr RecFilter::operator()(Var x)
{
	return RecFilterRefExpr(*this, { Expr(x) });
}
RecFilterRefExpr RecFilter::operator()(Var x, Var y)
{
	return RecFilterRefExpr(*this, { Expr(x), Expr(y) });
}
RecFilterRefExpr RecFilter::operator()(Var x, Var y, Var z)
{
	return RecFilterRefExpr(*this, { Expr(x), Expr(y), Expr(z) });
}
RecFilterRefExpr RecFilter::operator()(vector<Var> x)
{
	vector<Expr> x_expr(x.size());
	for (int i = 0; i < x.size(); i++)
	{
		x_expr[i] = Expr(x[i]);
	}
	return RecFilterRefExpr(*this, x_expr);
}
RecFilterRefExpr RecFilter::operator()(Expr x)
{
	return RecFilterRefExpr(*this, { x });
}
RecFilterRefExpr RecFilter::operator()(Expr x, Expr y)
{
	return RecFilterRefExpr(*this, { x, y });
}
RecFilterRefExpr RecFilter::operator()(Expr x, Expr y, Expr z)
{
	return RecFilterRefExpr(*this, { x, y, z });
}
RecFilterRefExpr RecFilter::operator()(vector<Expr> x)
{
	return RecFilterRefExpr(*this, x);
}

void RecFilter::define(vector<RecFilterDim> pure_args, vector<Expr> pure_def)
{
	assert(contents.get());
	assert(!pure_args.empty());
	assert(!pure_def.empty());

	contents.get()->type = pure_def[0].type();
	for (int i = 1; i < pure_def.size(); i++)
	{
		if (contents.get()->type != pure_def[i].type())
		{
			cerr << "Type of all Tuple elements in filter definition must be same" << endl;
			assert(false);
		}
	}

	if (!contents.get()->filter_info.empty())
	{
		cerr << "Recursive filter " << contents.get()->name << " already defined" << endl;
		assert(false);
	}

	RecFilterFunc rf;
	rf.func = Function(contents.get()->name);
	rf.func_category = INTRA_N;

	// add the arguments

	for (int i = 0; i < pure_args.size(); i++)
	{
		FilterInfo s;

		// set the variable and filter dimension
		s.filter_dim = i;
		s.var = pure_args[i].var();

		// extent and domain of all scans in this dimension
		s.image_width = pure_args[i].num_pixels();
		s.tile_width = s.image_width;
		s.rdom = RDom(0, s.image_width, unique_name("r" + s.var.name()));

		// default values for now
		s.num_scans = 0;
		s.feedback_order = 0;
		s.feedfwd_order = 0;

		contents.get()->filter_info.push_back(s);

		// add tag the dimension as pure
		rf.pure_var_category.insert(make_pair(pure_args[i].var().name(), VarTag(FULL, i)));
	}

	contents.get()->func.insert(make_pair(rf.func.name(), rf));

	// add the right hand side definition
	Function f = rf.func;

	vector<string> args;
	for (int i = 0; i < contents.get()->filter_info.size(); i++)
	{
		args.push_back(contents.get()->filter_info[i].var.name());
	}
	f.define(args, pure_def);
}

// -----------------------------------------------------------------------------

void RecFilter::set_clamped_image_border(void)
{
	if (!contents.get()->filter_info.empty())
	{
		cerr << "Recursive filter " << contents.get()->name << " already defined" << endl;
		assert(false);
	}
	contents.get()->clamped_border = true;
}

void RecFilter::add_filter(RecFilterDim x, vector<float> coeff, const int feedfwd_size)
{
	add_filter(RecFilterDimAndCausality(x, true), coeff, feedfwd_size);
}

void RecFilter::add_filter(RecFilterDimAndCausality x, vector<float> coeff, const int feedfwd_size)
{
	RecFilterFunc& rf = internal_function(contents.get()->name);
	Function        f = rf.func;

	if (!f.has_pure_definition())
	{
		cerr << "Cannot add scans to recursive filter " << f.name()
			<< " before specifying an initial definition using RecFilter::define()" << endl;
		assert(false);
	}

	vector<float> feedfwd_vec(0), feedback_vec(0);
	feedfwd_vec.insert(feedfwd_vec.begin(), coeff.begin(), coeff.begin() + feedfwd_size);
	feedback_vec.insert(feedback_vec.begin(), coeff.begin() + feedfwd_size, coeff.end());

	bool causal = x.causal();

	// filter order and csausality
	int fwd_order = feedfwd_vec.size(), back_order = feedback_vec.size();

	// image dimension for the scan
	int dimension = -1;
	for (int i = 0; dimension < 0 && i < f.args().size(); i++)
	{
		if (f.args()[i] == x.var().name())
		{
			dimension = i;
		}
	}
	if (dimension == -1)
	{
		cerr << "Variable " << x << " is not one of the dimensions of the "
			<< "recursive filter " << f.name() << endl;
		assert(false);
	}

	// reduction domain for the scan
	RDom rx = contents.get()->filter_info[dimension].rdom;
	Expr width = contents.get()->filter_info[dimension].image_width;

	// create the LHS args, replace x by rx for causal and
	// x by w-1-rx for anticausal
	vector<Expr> args;
	for (int i = 0; i < f.args().size(); i++)
	{
		if (i == dimension)
		{
			if (causal)
			{
				args.push_back(rx);
			}
			else
			{
				args.push_back(width - 1 - rx);
			}
		}
		else
		{
			args.push_back(Var(f.args()[i]));
		}
	}


	// RHS scan definition
	vector<Expr> values(f.values().size());
	for (int i = 0; i < values.size(); i++)
	{
		/*values[i] = Cast::make(contents.get()->type,feedfwd) *
			Call::make(f, args, i);*/

		values[i] = make_zero(contents.get()->type);

		for (int j = 0; j < feedfwd_vec.size(); j++)
		{
			vector<Expr> call_args = args;
			Expr substitued;

			if (causal)
			{
				substitued = substitute(f.args()[dimension], max(call_args[dimension] - j, 0), f.values()[i]);
				//call_args[dimension] = max(call_args[dimension] - (j + 1), 0);
			}
			else
			{
				substitued = substitute(f.args()[dimension], min(call_args[dimension] + j, width - 1), f.values()[i]);
				//call_args[dimension] = min(call_args[dimension] + (j + 1), width - 1);
			}
			values[i] += Cast::make(contents.get()->type, feedfwd_vec[j]) *
				select(rx >= j, substitued, make_zero(contents.get()->type));
		}

		for (int j = 0; j < feedback_vec.size(); j++)
		{
			vector<Expr> call_args = args;
			if (causal)
			{
				call_args[dimension] = max(call_args[dimension] - (j + 1), 0);
			}
			else
			{
				call_args[dimension] = min(call_args[dimension] + (j + 1), width - 1);
			}
			if (contents.get()->clamped_border)
			{
				values[i] += Cast::make(contents.get()->type, feedback_vec[j]) *
					Call::make(f, call_args, i);
			}
			else
			{
				values[i] += Cast::make(contents.get()->type, feedback_vec[j]) *
					select(rx > j, Call::make(f, call_args, i), make_zero(contents.get()->type));
			}
		}
	}
	f.define_update(args, values);

	// add details to the split info struct
	FilterInfo s = contents.get()->filter_info[dimension];
	s.scan_id.insert(s.scan_id.begin(), f.updates().size() - 1);
	s.scan_causal.insert(s.scan_causal.begin(), causal);
	s.num_scans = s.num_scans + 1;
	//s.filter_order = std::max(s.filter_order, scan_order)f;
	s.feedback_order = feedback_vec.size();
	s.feedfwd_order = feedfwd_vec.size();
	/*s.tol = this->tol;
	s.max_iter = this->max_iter;*/
	contents.get()->filter_info[dimension] = s;

	// copy all the existing feedback/feedfwd coeff to the new arrays
	// add the coeff of the newly added scan as the last row of coeff
	int num_scans = f.updates().size();
	int max_order = contents.get()->feedfwd_coeff.height();
	Buffer<float> feedfwd_coeff(num_scans, std::max(max_order, int(feedfwd_vec.size())));
	max_order = contents.get()->feedback_coeff.height();
	Buffer<float> feedback_coeff(num_scans, std::max(max_order, int(feedback_vec.size())));
	for (int j = 0; j < num_scans - 1; j++)
	{
		for (int i = 0; i < contents.get()->feedback_coeff.height(); i++)
		{
			feedfwd_coeff(j, i) = contents.get()->feedfwd_coeff(j, i);
		}
		for (int i = 0; i < contents.get()->feedback_coeff.height(); i++)
		{
			feedback_coeff(j, i) = contents.get()->feedback_coeff(j, i);
		}
	}
	for (int i = 0; i < feedfwd_vec.size(); i++)
	{
		feedfwd_coeff(num_scans - 1, i) = feedfwd_vec[i];
	}
	for (int i = 0; i < feedback_vec.size(); i++)
	{
		feedback_coeff(num_scans - 1, i) = feedback_vec[i];
	}

	// update the feedback and feedforward coeff matrices in all filter info
	contents.get()->feedfwd_coeff = feedfwd_coeff;
	contents.get()->feedback_coeff = feedback_coeff;

	// copy the dimension tags from pure def replacing x by rx
	// change the function tag from pure to scan
	map<string, VarTag> update_var_category = rf.pure_var_category;

	// decrement the full count of all vars whose count was more than count of x
	int count_x = update_var_category[x.var().name()].count();
	update_var_category.erase(x.var().name());
	map<string, VarTag>::iterator vit;
	for (vit = update_var_category.begin(); vit != update_var_category.end(); vit++)
	{
		if (vit->second.check(FULL) && !vit->second.check(SCAN))
		{
			int count = vit->second.count();
			if (count > count_x)
			{
				update_var_category[vit->first] = VarTag(FULL, count - 1);
			}
		}
	}
	update_var_category.insert(make_pair(rx.x.name(), FULL | SCAN));
	rf.update_var_category.push_back(update_var_category);
}

// Expr係数のadd_filter
void RecFilter::add_filter(RecFilterDimAndCausality x, std::vector<Halide::Expr> coeff, const int feedfwd_size)
{
	RecFilterFunc& rf = internal_function(contents.get()->name);
	Function        f = rf.func;

	Algorithm |= Custom; // CustomRecFilter限定にする

	if (!f.has_pure_definition())
	{
		cerr << "Cannot add scans to recursive filter " << f.name()
			<< " before specifying an initial definition using RecFilter::define()" << endl;
		assert(false);
	}

	vector<Expr> feedfwd_vec(0), feedback_vec(0);
	feedfwd_vec.insert(feedfwd_vec.begin(), coeff.begin(), coeff.begin() + feedfwd_size);
	feedback_vec.insert(feedback_vec.begin(), coeff.begin() + feedfwd_size, coeff.end());

	bool causal = x.causal();

	// filter order and csausality
	int scan_order = std::max(feedfwd_vec.size(), feedback_vec.size());

	// image dimension for the scan
	int dimension = -1;
	for (int i = 0; dimension < 0 && i < f.args().size(); i++)
	{
		if (f.args()[i] == x.var().name())
		{
			dimension = i;
		}
	}
	if (dimension == -1)
	{
		cerr << "Variable " << x << " is not one of the dimensions of the "
			<< "recursive filter " << f.name() << endl;
		assert(false);
	}

	// reduction domain for the scan
	RDom rx = contents.get()->filter_info[dimension].rdom;
	Expr width = contents.get()->filter_info[dimension].image_width;

	// create the LHS args, replace x by rx for causal and
	// x by w-1-rx for anticausal
	vector<Expr> args;
	for (int i = 0; i < f.args().size(); i++)
	{
		if (i == dimension)
		{
			if (causal)
			{
				args.push_back(rx);
			}
			else
			{
				args.push_back(width - 1 - rx);
			}
		}
		else
		{
			args.push_back(Var(f.args()[i]));
		}
	}


	// RHS scan definition
	vector<Expr> values(f.values().size());
	for (int i = 0; i < values.size(); i++)
	{
		/*values[i] = Cast::make(contents.get()->type,feedfwd) *
			Call::make(f, args, i);*/

		values[i] = make_zero(contents.get()->type);

		for (int j = 0; j < feedfwd_vec.size(); j++)
		{
			vector<Expr> call_args = args;
			Expr substitued;

			if (causal)
			{
				substitued = substitute(f.args()[dimension], max(call_args[dimension] - j, 0), f.values()[i]);
				//call_args[dimension] = max(call_args[dimension] - (j + 1), 0);
			}
			else
			{
				substitued = substitute(f.args()[dimension], min(call_args[dimension] + j, width - 1), f.values()[i]);
				//call_args[dimension] = min(call_args[dimension] + (j + 1), width - 1);
			}
			values[i] += Cast::make(contents.get()->type, feedfwd_vec[j]) *
				select(rx >= j, substitued, make_zero(contents.get()->type));
		}

		for (int j = 0; j < feedback_vec.size(); j++)
		{
			vector<Expr> call_args = args;
			if (causal)
			{
				call_args[dimension] = max(call_args[dimension] - (j + 1), 0);
			}
			else
			{
				call_args[dimension] = min(call_args[dimension] + (j + 1), width - 1);
			}
			if (contents.get()->clamped_border)
			{
				values[i] += Cast::make(contents.get()->type, feedback_vec[j]) *
					Call::make(f, call_args, i);
			}
			else
			{
				values[i] += Cast::make(contents.get()->type, feedback_vec[j]) *
					select(rx > j, Call::make(f, call_args, i), make_zero(contents.get()->type));
			}
		}
	}

	// valuesのxを置換
	for (int i = 0; i < values.size(); i++)
	{
		if (expr_uses_var(values[i], x.var().name()))
		{
			values[i] = substitute(x.var(), args[dimension], values[i]);
		}
	}

	f.define_update(args, values);

	// add details to the split info struct
	FilterInfo s = contents.get()->filter_info[dimension];
	s.scan_id.insert(s.scan_id.begin(), f.updates().size() - 1);
	s.scan_causal.insert(s.scan_causal.begin(), causal);
	s.num_scans = s.num_scans + 1;
	//s.filter_order = std::max(s.filter_order, scan_order);
	s.feedback_order = feedback_vec.size();
	s.feedfwd_order = feedfwd_vec.size();
	contents.get()->filter_info[dimension] = s;

	// copy all the existing feedback/feedfwd coeff to the new arrays
	// add the coeff of the newly added scan as the last row of coeff
	int num_scans = f.updates().size();
	//int max_order = contents.get()->feedfwd_coeff_expr.;
	//Buffer<float> feedfwd_coeff(num_scans, std::max(max_order, int(feedfwd_vec.size())));
	vector<vector<Expr>> feedfwd_coeff_expr(num_scans);
	//max_order = contents.get()->feedback_coeff.height();
	//Buffer<float> feedback_coeff(num_scans, std::max(max_order, int(feedback_vec.size())));
	vector<vector<Expr>> feedback_coeff_expr(num_scans);

	// coeff情報を前のからもってくる
	for (int j = 0; j < num_scans - 1; j++)
	{
		auto temp_fwd = contents.get()->feedfwd_coeff_expr[j];
		auto temp_back = contents.get()->feedback_coeff_expr[j];
		if (temp_fwd.empty() || temp_back.empty()) continue;
		feedfwd_coeff_expr[j] = temp_fwd;
		feedback_coeff_expr[j] = temp_back;
	}

	feedfwd_coeff_expr[num_scans - 1] = feedfwd_vec;
	feedback_coeff_expr[num_scans - 1] = feedback_vec;

	//for (int i = 0; i < feedfwd_size; i++)
	//{
	//	feedfwd_coeff(num_scans - 1, i) = feedfwd_vec[i];
	//}
	//for (int i = 0; i < scan_order; i++)
	//{
	//	//feedback_coeff(num_scans - 1, i) = feedback_vec[i];
	//}

	// update the feedback and feedforward coeff matrices in all filter info
	contents.get()->feedfwd_coeff_expr = feedfwd_coeff_expr;
	contents.get()->feedback_coeff_expr = feedback_coeff_expr;

	// copy the dimension tags from pure def replacing x by rx
	// change the function tag from pure to scan
	map<string, VarTag> update_var_category = rf.pure_var_category;

	// decrement the full count of all vars whose count was more than count of x
	int count_x = update_var_category[x.var().name()].count();
	update_var_category.erase(x.var().name());
	map<string, VarTag>::iterator vit;
	for (vit = update_var_category.begin(); vit != update_var_category.end(); vit++)
	{
		if (vit->second.check(FULL) && !vit->second.check(SCAN))
		{
			int count = vit->second.count();
			if (count > count_x)
			{
				update_var_category[vit->first] = VarTag(FULL, count - 1);
			}
		}
	}
	update_var_category.insert(make_pair(rx.x.name(), FULL | SCAN));
	rf.update_var_category.push_back(update_var_category);
}

// -----------------------------------------------------------------------------

RecFilterSchedule RecFilter::full_schedule(void)
{
	if (contents.get()->tiled)
	{
		cerr << "Filter is tiled, use RecFilter::intra_schedule() "
			<< "and RecFilter::inter_schedule()\n" << endl;
		assert(false);
	}
	return RecFilterSchedule(*this, { name() });
}

RecFilterSchedule RecFilter::intra_schedule(int id)
{
	if (!contents.get()->tiled)
	{
		cerr << "\nNo intra-tile terms to schedule in a non-tiled filter" << endl;
		cerr << "Use RecFilter::schedule() and Halide scheduling API\n" << endl;
		assert(false);
	}

	vector<string> func_list;

	map<string, RecFilterFunc>::iterator f_it = contents.get()->func.begin();
	for (; f_it != contents.get()->func.end(); f_it++)
	{
		bool function_condition = false;
		FuncTag ftag = f_it->second.func_category;

		switch (id)
		{
		case 0: function_condition |= (ftag == FuncTag(INTRA_1) | ftag == FuncTag(INTRA_N)); break;
		case 1: function_condition |= (ftag == FuncTag(INTRA_N)); break;
		default:function_condition |= (ftag == FuncTag(INTRA_1)); break;
		}

		if (function_condition)
		{
			string func_name = f_it->second.func.name();

			// all functions which are REINDEX and call/called by this function
			map<string, RecFilterFunc>::iterator g_it = contents.get()->func.begin();
			for (; g_it != contents.get()->func.end(); g_it++)
			{
				RecFilterFunc rf = g_it->second;
				if (!g_it->second.pure_schedule.empty()) continue;
				if (rf.func_category == REINDEX)
				{
					if (rf.producer_func == func_name || rf.consumer_func == func_name)
					{
						func_list.push_back(g_it->first);
					}
				}
			}
			func_list.push_back(func_name);
		}
	}

	if (func_list.empty())
	{
		cerr << "Warning: No " << (id == 0 ? " " : (id == 1 ? "1D " : "nD "));
		cerr << "intra tile functions to schedule" << endl;
	}
	return RecFilterSchedule(*this, func_list);
}

RecFilterSchedule RecFilter::inter_schedule(void)
{
	if (!contents.get()->tiled)
	{
		cerr << "\nNo inter-tile terms to schedule in a non-tiled filter" << endl;
		cerr << "Use RecFilter::schedule() and Halide scheduling API\n" << endl;
		assert(false);
	}

	vector<string> func_list;

	map<string, RecFilterFunc>::iterator f_it = contents.get()->func.begin();
	for (; f_it != contents.get()->func.end(); f_it++)
	{
		if (f_it->second.func_category == INTER)
		{
			string func_name = f_it->second.func.name();
			func_list.push_back(func_name);
		}
	}

	if (func_list.empty())
	{
		//cerr << "Warning: No inter tile functions to schedule" << endl; // 出ても問題なし
	}

	return RecFilterSchedule(*this, func_list);
}

void RecFilter::compute_at(RecFilter& external)
{
	// ***** memo *****
	// externalのFinalにcompute_atすると，初期化とフィルタで再計算する
	// Finalではなくスキン関数にcompute_atを指定すればOK
	Var external_var;

	// find the innermost tile index to use as loop level
	// GPU targets - this is CUDA tile index x
	// CPU targets - this is first OUTER index
	if (target().has_gpu_feature())
	{
		//external_var = Var::gpu_blocks(); // koko
		std::cout << "hofe\n";
	}
	else
	{
		bool found = false;
		map<string, VarTag> pure_var_category = external.internal_function(external.name()).pure_var_category;
		map<string, VarTag>::iterator vit = pure_var_category.begin();
		map<string, VarTag>::iterator vend = pure_var_category.end();
		for (; !found && vit != vend; vit++)
		{
			if (vit->second == VarTag(OUTER, 0))
			{
				found = true;
				external_var = Var(vit->first);
			}
			//std::cout << vit->first << " : " << vit->second << std::endl;
		}

		if (!found)
		{
			cerr << name() << " cannot be computed locally in another recursive "
				<< "filter because the external filter does not a tile index where "
				<< name() << " can be computed, possibly because it is not tiled"
				<< endl;
			assert(false);
		}
	}

	Func temp_f(external.as_func().function());
	compute_at(temp_f, external_var);
	//compute_at(external.as_func(), external_var);
}

void RecFilter::store_at(RecFilter external)
{
	Var external_var;
	Func external_func = external.as_func();
	RecFilterFunc external_rfunc = external.internal_function(external.name());

	// find the innermost tile index to use as loop level
	// GPU targets - this is CUDA tile index x
	// CPU targets - this is first OUTER index
	if (target().has_gpu_feature())
	{
		//external_var = Var::gpu_blocks(); // koko
		std::cout << "hofe\n";
	}
	else
	{
		bool found = false;
		map<string, VarTag> pure_var_category = external_rfunc.pure_var_category;
		map<string, VarTag>::iterator vit = pure_var_category.begin();
		map<string, VarTag>::iterator vend = pure_var_category.end();
		for (; !found && vit != vend; vit++)
		{
			if (vit->second == VarTag(INNER, 0))
			{
				found = true;
				external_var = Var(vit->first);
			}
		}

		if (!found)
		{
			cerr << name() << " cannot be stored locally in another recursive "
				<< "filter because the external filter does not a tile index where "
				<< name() << " can be computed, possibly because it is not tiled"
				<< endl;
			assert(false);
		}
	}

	store_at(external_func, external_var);
}

void RecFilter::compute_root()
{
	RecFilterFunc rF = internal_function(name());
	Function F = rF.func;
	Func(F).compute_root();
	rF.pure_schedule.push_back("compute_root()");
}

void RecFilter::compute_with(RecFilter external)
{

	if (this->Algorithm != external.Algorithm)
	{
		cerr << "can not compute_with(" << external.name() << "), because it was defined in different algorithm.\n";
		assert(false);
	}

	Var external_var;
	RecFilterFunc external_rfunc = external.internal_function(external.name());

	// find the innermost tile index to use as loop level
	// GPU targets - this is CUDA tile index x
	// CPU targets - this is first OUTER index
	if (target().has_gpu_feature())
	{
		//external_var = Var::gpu_blocks(); // koko
		std::cout << "hofe\n";
	}
	else
	{
		bool found = false;
		map<string, VarTag> pure_var_category = external_rfunc.pure_var_category;
		map<string, VarTag>::iterator vit = pure_var_category.begin();
		map<string, VarTag>::iterator vend = pure_var_category.end();

		for (; !found && vit != vend; vit++)
		{
			if (vit->second == VarTag(INNER, 0))
			{
				found = true;
				external_var = Var(vit->first);
			}
		}

		if (!found)
		{
			cerr << name() << " cannot be computed locally in another recursive "
				<< "filter because the external filter does not a tile index where "
				<< name() << " can be computed, possibly because it is not tiled"
				<< endl;
			assert(false);
		}
	}
	map<string, RecFilterFunc> external_funcs = external.contents->func,
		ref_funcs = this->contents->func;

	map<string, RecFilterFunc>::iterator external_it = external_funcs.begin();
	map<string, RecFilterFunc>::iterator ref_it = ref_funcs.begin();

	for (; external_it != external_funcs.end(); external_it++, ref_it++)
	{
		if (ref_it->first == name() && is_inlined) continue; // inline展開されてcompute_atしている場合はcompute_withできない

		// 内部でcompute_atしている場合は，外部の関数でcompute_withできない(むりやり)
		// -> というかfinal以外はできない
		if (ref_it->second.func_type != FINAL)
		{
			continue;
		}

		RecFilterFunc external = external_it->second;
		RecFilterFunc ref = ref_it->second;
		Function eF = external.func;
		Function rF = ref.func;
		Func(rF).compute_with(Func(eF), external_var);

		stringstream s;
		s << "comput_with(" << eF.name() << ", " << external_var.name() << ")";
		internal_function(ref_it->first).pure_schedule.push_back(s.str());

		for (int i = 0; i < rF.updates().size(); i++)
		{
			bool found = false;
			map<string, VarTag> update_var_category = external.update_var_category[i];
			map<string, VarTag>::iterator vit = update_var_category.begin();
			map<string, VarTag>::iterator vend = update_var_category.end();
			for (; !found && vit != vend; vit++)
			{
				if (vit->second == VarTag(INNER, 0))
				{
					found = true;
					external_var = Var(vit->first);
				}
			}

			if (!found)
			{
				cerr << name() << " cannot be computed locally in another recursive "
					<< "filter because the external filter does not a tile index where "
					<< name() << " can be computed, possibly because it is not tiled"
					<< endl;
				assert(false);
			}


			Func(rF).update(i).compute_with(Func(eF).update(i), external_var);

			stringstream us;
			us << "comput_with(" << eF.name() << ".update(" << i << "), " << external_var.name() << ")";
			internal_function(ref_it->first).update_schedule.at(i).push_back(us.str());
		}
	}
}

void RecFilter::compute_at(Func& external, Var looplevel)
{
	// Func representing the final result
	RecFilterFunc& rF = internal_function(name());
	Function f = rF.func;

	// check that the filter does not depend upon F
	if (contents.get()->func.find(external.name()) != contents.get()->func.end())
	{
		cerr << "Cannot compute " << name() << " at " << external.name()
			<< " because it is a consumer of " << external.name() << endl;
		assert(false);
	}

	// this function must not have a consumer because this is now being set to
	// be computed at something else
	if (!rF.consumer_func.empty() || rF.external_consumer_func.defined())
	{
		cerr << "Cannot compute " << name() << " at " << external.name()
			<< " because it already has a consumer " << endl;
		assert(false);
	}

	f.lock_loop_levels();

	// check that the compute looplevel of the final result is not already set
	if (!f.schedule().compute_level().is_inlined() ||
		!f.schedule().store_level().is_inlined())
	{
		cerr << "Cannot compute " << name() << " inside " << external.name()
			<< " because it is set to be computed at "
			<< f.schedule().compute_level().func() << " "
			<< f.schedule().compute_level().var().name() << endl;
		assert(false);
	}

	// new compute at level
	string compute_level_str = "compute_at(" + external.name() +
		", Var(\"" + looplevel.name() + "\"))";

	rF.external_consumer_func = external;
	rF.external_consumer_var = looplevel;

	// 外部の関数でcompute_atするなら，RecFilterの"皮"になっている関数はインライン展開したほうが，
	// 余計な関数を挟まないので速くなるはず．
	Function temp(f.name() + "_inlined");
	temp.define(f.args(), f.values());
	Func(temp).compute_inline();
	//rF.func.schedule() = temp.schedule();
	rF.pure_schedule.clear();
	rF.pure_schedule.push_back("compute_inline()");
	std::map<std::string, Halide::Internal::Function> env;
	Halide::Internal::populate_environment(external.function(), env);
	// external先の全ての関数でインライン化関数を呼ぶように
	for (auto it = env.begin(); it != env.end(); it++)
	{
		it->second.substitute_calls(rF.func, temp); // external先がインライン化したのを呼びだすように
	}
	rF.func = temp;

	is_inlined = true;
	// set the producer of this function to be computed at the same looplevel

	for (auto it = contents.get()->func.begin(); it != contents.get()->func.end(); it++)
	{
		if (it->second.consumer_func == name())
		{
			RecFilterFunc& producer = internal_function(it->first);
			producer.func.lock_loop_levels(); // 追記
			//if (!producer.func.schedule().compute_level().is_inlined() || !producer.func.schedule().store_level().is_inlined())
			{
				Func(producer.func).compute_at(external, looplevel);
				producer.external_consumer_func = external;
				producer.external_consumer_var = looplevel;
				producer.pure_schedule.push_back(compute_level_str);
			}
			// find all Functions whose consumer is this the above producer and set them
			// to be computed inside the external function. This is because the Func
			// computing the final result is now being computed inside some loop of an
			// external function. So all upstream functions should be same, or else they
			// will trigger a write to global memory
			map<string, RecFilterFunc>::iterator fit;
			for (fit = contents.get()->func.begin(); fit != contents.get()->func.end(); fit++)
			{
				RecFilterFunc& rG = fit->second;
				if (rG.consumer_func == producer.func.name())
				{
					rG.external_consumer_func = external;
					rG.external_consumer_var = looplevel;
				}
			}
		}
	}
}

void RecFilter::store_at(Func external, Var looplevel)
{
	return;
}

void RecFilter::store_root()
{
	RecFilterFunc& rF = internal_function(name());
	Function f = rF.func;

	string store_level_str = "store_root()";

	Func(f).store_root();
	rF.pure_schedule.push_back(store_level_str);
}

void RecFilter::trace_all_realizations()
{
	std::vector<std::string> functions;
	std::map<std::string, RecFilterFunc>::iterator it = contents->func.begin();
	std::map<std::string, RecFilterFunc>::iterator end_it = contents->func.end();
	for (; it != end_it; it++)
	{
		Function f = it->second.func;
		Func(f).trace_realizations();
	}
}

void RecFilter::store_in(Halide::MemoryType memory_type)
{
	std::vector<std::string> functions;
	std::map<std::string, RecFilterFunc>::iterator it = contents->func.begin();
	std::map<std::string, RecFilterFunc>::iterator end_it = contents->func.end();
	string mt_str = "";
	if (memory_type == MemoryType::Auto)mt_str = "AUTO";
	else if (memory_type == MemoryType::Heap) mt_str = "Heap";
	else if (memory_type == MemoryType::Stack) mt_str = "Stack";
	else if (memory_type == MemoryType::Register) mt_str = "Register";
	else if (memory_type == MemoryType::GPUShared) mt_str = "GPUShared";
	else if (memory_type == MemoryType::LockedCache) mt_str = "LockedCache";
	else if (memory_type == MemoryType::VTCM) mt_str = "VTCM";
	for (; it != end_it; it++)
	{
		Function f = it->second.func;
		Func(f).store_in(memory_type);
		it->second.pure_schedule.push_back("store_in(" + mt_str + ")");
	}
}

// -----------------------------------------------------------------------------

void RecFilter::cpu_auto_schedule(void)
{
	if (contents.get()->tiled)
	{
		if (
			Algorithm == (Custom | SlidingSingle)
			|| Algorithm == (Custom | SlidingSinglePostProcess)
			|| Algorithm == (Custom | SlidingSingleMultiDelta)
			|| Algorithm == (Custom | SlidingSingleMultiDeltaPostProcess)
			)
		{
			if (slidingScheduled)
			{
				cpu_auto_schedule_for_sliding_custom();
			}
			else
			{
				cpu_auto_schedule_for_sliding();
			}
		}
		else
		{
			cpu_auto_intra_schedule();
			cpu_auto_inter_schedule();
		}
	}
	else
	{
		cpu_auto_full_schedule();
	}
}

// ほんとは要探索
Var mapSchedulleTagToVar(ScheduleTag tag)
{
	switch (tag)
	{
	case XO: return Var("xo");
	case XI: return Var("xi");
	case YO: return Var("yo");
	case YI: return Var("yi");
	case C: return Var("c");
	}
}

VarTag RecFilter::mapScheduleTagToVarTag(ScheduleTag tag, bool isY)
{
	switch (tag)
	{
	case XO: return outer();
	case XI: return isY ? inner(0) : inner_scan();
	case YO: return outer();
	case YI: return isY ? inner_scan() : inner(0);
	case C: return inner_channels();
	}
}

void RecFilter::cpu_auto_schedule_for_sliding(void)
{
	if (!contents.get()->tiled)
	{
		cerr << "Filter is not tiled, use RecFilter::cpu_auto_full_schedule()\n" << endl;
		assert(false);
	}

	RecFilterSchedule R = intra_schedule(0);

	if (R.empty())
	{
		return;
	}

	int max_tile = 0;
	for (int i = 0; i < contents.get()->filter_info.size(); i++)
	{
		if (contents.get()->filter_info[i].tile_width < contents.get()->filter_info[i].image_width)
		{
			max_tile = std::max(max_tile, contents.get()->filter_info[i].tile_width);
		}
	}

	// よくわからないがinfo[1]にデータがちゃんと入ってたらyフィルタ, info[0]ならxフィルタ
	bool isYfilter = contents.get()->filter_info[1].feedfwd_order > 0;

	if (isYfilter)
	{
		// 各フィルタのスケジュール
		// 1. フィルタ方向にsplit
		// 2. xi, yi, xo,yo, cでreorder
		// 3. xiをベクトル化
		// 
		// メモリ的に連続なx方向でベクトル化するのが最も速いためxiを最内部ループにしてベクトル化
		// yフィルタではunroll, paralellはしないほうが速い
		// split長=vectorize長, set_vectorization_witdhの値は使っていない
		R
			.split(full(0), max_tile, inner(), outer())
			.reorder({ inner(0),inner_scan(), outer(), outer_channels() })
			.vectorize(inner(0))
			.compute_locally(); // こいつを最後にしないと，内側のOUTERでcompute_atできない(元々一番上にあった)

		{
			// 内部のFuncを取り出す
			// func_listは各フィルタ
			// skinFuncは各フィルタの和を取るFuncでoutputそのもの
			vector<Func> func_list;
			RecFilterFunc skinFunc;

			map<string, RecFilterFunc>::iterator f_it = contents.get()->func.begin();
			for (; f_it != contents.get()->func.end(); f_it++)
			{
				if (f_it->second.func_type == SKIN)
				{
					skinFunc = f_it->second;
					continue;
				}

				FuncTag ftag = f_it->second.func_category;
				if (ftag == FuncTag(INTRA_1) | ftag == FuncTag(INTRA_N))
				{

					func_list.push_back(Func(f_it->second.func));
				}
			}

			if (func_list.empty())
			{
				cerr << "Warning: No intra tile functions to schedule" << endl;
			}

			// skinFuncのスケジューリング
			// 
			// フィルタ方向じゃないxでsplit, parallel, vectorize
			// フィルタ方向のyでparallelは遅くなるがxは速くなる
			// ↑でやった各フィルタのスケジューリングと同じreorder
			Var xo("xo"), xi("xi");
			Func(skinFunc.func)
				.split(Var("x"), xo, xi, 128)
				.reorder(xi, Var("yi"), xo, Var("yo"))
				//.parallel(Var("yo"))
				.parallel(Var("c"))
				//.parallel(xo)
				.vectorize(xi);

			// 各フィルタの追加スケジュール
			for (int k = 0; k < func_list.size(); k++)
			{
				// xi, yi, xo, yoの
				// yiでcompute_atすることで高速化 
				// yiでstoreすると、直近2要素が必要となるスライディング処理では不都合なのでxoでstore
				func_list[k]
					.compute_at(Func(skinFunc.func), Var("yi"))
					.store_at(Func(skinFunc.func), Var("xo"));

				// 各フィルタのループを1回のループに
				// xi, yi, xo, yoのxiをcompute_with
				// RecFilterではdefinitionはundef, update(0)でフィルタが定義されている
				if (k != func_list.size() - 1)
				{
					func_list[k].update(0).compute_with(func_list[k + 1].update(0), RVar("xi"));
				}
			}
		}
	}
	else
	{
		// 各フィルタのスケジュール
		// 1. フィルタ方向にsplit
		// 2. yi, xi, xo, yoでreorder
		// 3. yiをベクトル化
		// 
		// メモリ的に連続なx方向にベクトル化したいが、スライディング処理はフィルタ方向にベクトル化できないためyiでベクトル化
		// xフィルタでは無指定unrollすると速くなる
		// yフィルタではx parallelすると速くなる
		// split長=vectorize長, set_vectorization_witdhの値は使っていない
		R
			.split(full(0), max_tile, inner(), outer())
			.reorder({ inner(0),inner_scan(), outer(), outer_channels() })
			.vectorize(inner(0))
			.unroll(inner_scan())
			.compute_locally(); // こいつを最後にしないと，内側のOUTERでcompute_atできない(元々一番上にあった)

		//if (parallel)
			//R.parallel(outer());


		{
			// 内部のFuncを取り出す
			// func_listは各フィルタ
			// skinFuncは各フィルタの和を取るFuncでoutputそのもの
			vector<Func> func_list;
			RecFilterFunc skinFunc;

			map<string, RecFilterFunc>::iterator f_it = contents.get()->func.begin();
			for (; f_it != contents.get()->func.end(); f_it++)
			{
				if (f_it->second.func_type == SKIN)
				{
					skinFunc = f_it->second;
					continue;
				}

				FuncTag ftag = f_it->second.func_category;
				if (ftag == FuncTag(INTRA_1) | ftag == FuncTag(INTRA_N))
				{
					func_list.push_back(Func(f_it->second.func));
				}
			}

			if (func_list.empty())
			{
				cerr << "Warning: No intra tile functions to schedule" << endl;
			}

			Var outer;
			map<string, VarTag> pure_var_category = skinFunc.pure_var_category;
			map<string, VarTag>::iterator vit = pure_var_category.begin();
			map<string, VarTag>::iterator vend = pure_var_category.end();

			for (; vit != vend; vit++)
			{
				if (vit->second == VarTag(OUTER, 0))
				{
					outer = Var(vit->first);
				}
			}

			// skinFuncのスケジューリング
			// 
			// フィルタ方向じゃないyでsplit, parallel
			// vectorizeはy方向なのでしないほうが早い
			// reoderはしてもしなくても変わらない
			Func(skinFunc.func)
				.split(Var("y"), Var("yo"), Var("yi"), 128)
				.parallel(Var("c"))
				.parallel(Var("yo"));

			// 各フィルタの追加スケジュール
			for (int k = 0; k < func_list.size(); k++)
			{
				// yi, xi, xo, yoの
				// xoでcompute_atすることで高速化
				// 
				// yフィルタのようにstoreとcomputeを分けても速度は変わらなかった、恐らくベクトル化がyiでメモリ的に最適ではないのが原因
				func_list[k]
					.compute_at(Func(skinFunc.func), outer)
					.store_at(Func(skinFunc.func), outer);

				// 各フィルタのループを1回のループに
				// yi, xi, xo, yoのyiをcompute_with
				// RecFilterではdefinitionはundef, update(0)でフィルタが定義されている
				if (k != func_list.size() - 1)
				{
					func_list[k].update(0).compute_with(func_list[k + 1].update(0), RVar("yi"));
				}
			}
		}
	}
}

void RecFilter::cpu_auto_schedule_for_sliding_custom(void)
{
	if (!contents.get()->tiled)
	{
		cerr << "Filter is not tiled, use RecFilter::cpu_auto_full_schedule()\n" << endl;
		assert(false);
	}

	RecFilterSchedule R = intra_schedule(0);

	if (R.empty())
	{
		return;
	}

	int max_tile = 0;
	for (int i = 0; i < contents.get()->filter_info.size(); i++)
	{
		if (contents.get()->filter_info[i].tile_width < contents.get()->filter_info[i].image_width)
		{
			max_tile = std::max(max_tile, contents.get()->filter_info[i].tile_width);
		}
	}

	// よくわからないがinfo[1]にデータがちゃんと入ってたらyフィルタ, info[0]ならxフィルタ
	bool isYfilter = contents.get()->filter_info[1].feedfwd_order > 0;

	if (isYfilter)
	{
		bool XIisInnerMost = schedule.reorder[0] == XI;
		if (XIisInnerMost)
		{
			R.split(full(0), schedule.splitX, inner(), outer());
		}

		// reorder
		vector<VarTag> varTag;
		for (ScheduleTag tag : schedule.reorder)
		{
			varTag.push_back(mapScheduleTagToVarTag(tag, true));
		}
		R.reorder(varTag);

		// vectorize
		for (std::pair<ScheduleTag, int> pair : schedule.vectorize)
		{
			R.vectorize(mapScheduleTagToVarTag(pair.first, true));
		}

		// parallel
		for (ScheduleTag tag : schedule.parallel)
		{
			R.parallel(mapScheduleTagToVarTag(tag, true));
		}

		// unroll
		for (std::pair<ScheduleTag, int> pair : schedule.unroll)
		{
			R.unroll(mapScheduleTagToVarTag(pair.first, true));
		}

		R.compute_locally();

		{
			// 内部のFuncを取り出す
			// func_listは各フィルタ
			// skinFuncは各フィルタの和を取るFuncでoutputそのもの
			vector<Func> func_list;
			RecFilterFunc skinFunc;

			map<string, RecFilterFunc>::iterator f_it = contents.get()->func.begin();
			for (; f_it != contents.get()->func.end(); f_it++)
			{
				if (f_it->second.func_type == SKIN)
				{
					skinFunc = f_it->second;
					continue;
				}

				FuncTag ftag = f_it->second.func_category;
				if (ftag == FuncTag(INTRA_1) | ftag == FuncTag(INTRA_N))
				{

					func_list.push_back(Func(f_it->second.func));
				}
			}

			if (func_list.empty())
			{
				cerr << "Warning: No intra tile functions to schedule" << endl;
			}

			// split
			Var xo("xo"), xi("xi");
			Func(skinFunc.func)
				.split(Var("x"), xo, xi, schedule.splitX);

			// reorder
			vector<VarOrRVar> reorderVar;
			for (ScheduleTag tag : schedule.reorder)
			{
				reorderVar.push_back(mapSchedulleTagToVar(tag));
			}
			Func(skinFunc.func).reorder(reorderVar);

			// vectorize
			for (std::pair<ScheduleTag, int> pair : schedule.vectorize)
			{
				if (pair.second == 0)
				{
					Func(skinFunc.func).vectorize(mapSchedulleTagToVar(pair.first));
				}
				else
				{
					Func(skinFunc.func).vectorize(mapSchedulleTagToVar(pair.first), pair.second);
				}
			}

			// parallel
			for (ScheduleTag tag : schedule.parallel)
			{
				Func(skinFunc.func).parallel(mapSchedulleTagToVar(tag));
			}

			// unroll
			for (std::pair<ScheduleTag, int> pair : schedule.unroll)
			{
				if (pair.second == 0)
				{
					Func(skinFunc.func).unroll(mapSchedulleTagToVar(pair.first));
				}
				else
				{
					Func(skinFunc.func).unroll(mapSchedulleTagToVar(pair.first), pair.second);
				}
			}

			// 各フィルタの追加スケジュール
			// ここは変えない
			for (int k = 0; k < func_list.size(); k++)
			{
				func_list[k]
					.compute_at(Func(skinFunc.func), XIisInnerMost ? Var("xo") : Var("xi"))
					;

				if (k != func_list.size() - 1)
				{
					func_list[k].update(0).compute_with(func_list[k + 1].update(0), XIisInnerMost ? RVar("xi") : RVar("ryi"));
				}
			}
		}
	}
	else
	{
		bool YIIsInnnermost = schedule.reorder[0] == YI;

		// split
		if (YIIsInnnermost)
		{
			R.split(full(0), schedule.splitX, inner(), outer());
		}

		// reorder
		vector<VarTag> varTag;
		for (ScheduleTag tag : schedule.reorder)
		{
			varTag.push_back(mapScheduleTagToVarTag(tag, false));
		}
		R.reorder(varTag);

		// vectorize
		for (std::pair<ScheduleTag, int> pair : schedule.vectorize)
		{
			R.vectorize(mapScheduleTagToVarTag(pair.first, false));
		}

		// parallel
		for (ScheduleTag tag : schedule.parallel)
		{
			R.parallel(mapScheduleTagToVarTag(tag, true));
		}

		// unroll
		for (std::pair<ScheduleTag, int> pair : schedule.unroll)
		{
			R.unroll(mapScheduleTagToVarTag(pair.first, false));
		}

		R.compute_locally();

		{
			// 内部のFuncを取り出す
			// func_listは各フィルタ
			// skinFuncは各フィルタの和を取るFuncでoutputそのもの
			vector<Func> func_list;
			RecFilterFunc skinFunc;

			map<string, RecFilterFunc>::iterator f_it = contents.get()->func.begin();
			for (; f_it != contents.get()->func.end(); f_it++)
			{
				if (f_it->second.func_type == SKIN)
				{
					skinFunc = f_it->second;
					continue;
				}

				FuncTag ftag = f_it->second.func_category;
				if (ftag == FuncTag(INTRA_1) | ftag == FuncTag(INTRA_N))
				{
					func_list.push_back(Func(f_it->second.func));
				}
			}

			if (func_list.empty())
			{
				cerr << "Warning: No intra tile functions to schedule" << endl;
			}

			// split
			Func(skinFunc.func)
				.split(Var("y"), Var("yo"), Var("yi"), schedule.splitY)
				;
			
			// reorder
			vector<VarOrRVar> reorderVar;
			for (ScheduleTag tag : schedule.reorder)
			{
				reorderVar.push_back(mapSchedulleTagToVar(tag));
			}
			Func(skinFunc.func).reorder(reorderVar);

			// vectorize
			for (std::pair<ScheduleTag, int> pair : schedule.vectorize)
			{
				if (pair.second == 0)
				{
					Func(skinFunc.func).vectorize(mapSchedulleTagToVar(pair.first));
				}
				else
				{
					Func(skinFunc.func).vectorize(mapSchedulleTagToVar(pair.first), pair.second);
				}
			}

			// parallel
			for (ScheduleTag tag : schedule.parallel)
			{
				Func(skinFunc.func).parallel(mapSchedulleTagToVar(tag));
			}

			// unroll
			//for (std::pair<ScheduleTag, int> pair : schedule.unroll)
			//{
			//	if (pair.second == 0)
			//	{
			//		Func(skinFunc.func).unroll(mapSchedulleTagToVar(pair.first));
			//	}
			//	else
			//	{
			//		Func(skinFunc.func).unroll(mapSchedulleTagToVar(pair.first), pair.second);
			//	}
			//}

			// 各フィルタの追加スケジュール
			for (int k = 0; k < func_list.size(); k++)
			{
				func_list[k]
					.compute_at(Func(skinFunc.func), YIIsInnnermost ? Var("xo") : Var("yi"));

				// 各フィルタのループを1回のループに
				// yi, xi, xo, yoのyiをcompute_with
				// RecFilterではdefinitionはundef, update(0)でフィルタが定義されている
				if (k != func_list.size() - 1)
				{
					func_list[k].update(0).compute_with(func_list[k + 1].update(0), YIIsInnnermost ? RVar("yi") : RVar("rxi"));
				}
			}
		}
	}
}

void RecFilter::cpu_auto_full_schedule(void)
{
	if (contents.get()->tiled)
	{
		cerr << "Filter is tiled, use RecFilter::cpu_auto_intra_schedule() "
			<< "and RecFilter::cpu_auto_inter_schedule()\n" << endl;
		assert(false);
	}

	int vector_width = RecFilter::vectorization_width;

	if (vector_width <= 0)
	{
		cerr << "Use RecFilter::set_vectorization_width() to specify "
			<< "the CPU target's native SSE width" << endl;
		assert(false);
	}

	// scan dimension can be unrolled
	// inner most dimension must be vectorized
	// all full dimensions must be parallelized

	full_schedule().compute_globally()
		.reorder(full_scan(), full())
		.vectorize(full(0), vector_width)
		.parallel(full());                  // TODO: only parallelize outermost, not all
}

void RecFilter::cpu_auto_intra_schedule(void)
{
	if (!contents.get()->tiled)
	{
		cerr << "Filter is not tiled, use RecFilter::cpu_auto_full_schedule()\n" << endl;
		assert(false);
	}

	int vector_width = RecFilter::vectorization_width;
	if (vector_width <= 0)
	{
		cerr << "Use RecFilter::set_vectorization_width() to specify "
			<< "the CPU target's native SSE width" << endl;
		assert(false);
	}

	RecFilterSchedule R = intra_schedule(0);

	if (R.empty())
	{
		return;
	}

	int max_tile = 0;
	for (int i = 0; i < contents.get()->filter_info.size(); i++)
	{
		if (contents.get()->filter_info[i].tile_width < contents.get()->filter_info[i].image_width)
		{
			max_tile = std::max(max_tile, contents.get()->filter_info[i].tile_width);
		}
	}

	// reorderについて，inner_scan(rxtとか)は2番目じゃない？一番内側はxiのベクトル化では？
	// -> rxt内側が一番早い．xiをベクトル化した状態でrxtをぶん回すから．
	R
		.split(full(0), max_tile, inner(), outer())    // convert upto 3 full dimensions
		.split(full(0), max_tile, inner(), outer())  // into tiles
		.split(full(0), max_tile, inner(), outer())
		.reorder({ inner_scan(), inner(), outer() })  // scan dimension is innermost こっちが元々 
		//.reorder({ inner(),inner_scan(), outer() })  // scan dimension is innermost 変更
		.vectorize(inner(0), vector_width)        // vectorize innermost non-scan dimension
		.compute_locally(); // こいつを最後にしないと，内側のOUTERでcompute_atできない(元々一番上にあった)

	if (parallel)
		R.parallel(outer());                          // TODO: only parallelize outermost
}

void RecFilter::cpu_auto_inter_schedule(void)
{
	if (!contents.get()->tiled)
	{
		cerr << "Filter is not tiled, use RecFilter::cpu_auto_full_schedule()\n" << endl;
		assert(false);
	}

	int vector_width = RecFilter::vectorization_width;
	if (vector_width <= 0)
	{
		cerr << "Use RecFilter::set_vectorization_width() to specify "
			<< "the CPU target's native SSE width" << endl;
		assert(false);
	}

	RecFilterSchedule R = inter_schedule();

	if (R.empty())
	{
		return;
	}

	int max_tile = 0;
	for (int i = 0; i < contents.get()->filter_info.size(); i++)
	{
		if (contents.get()->filter_info[i].tile_width < contents.get()->filter_info[i].image_width)
		{
			max_tile = std::max(max_tile, contents.get()->filter_info[i].tile_width);
		}
	}

	R.compute_globally()
		.split(full(0), max_tile, inner(), outer())         // convert upto 3 full dimensions
		.split(full(0), max_tile, inner(), outer())         // into tiles
		.split(full(0), max_tile, inner(), outer())
		.reorder({ outer_scan(), tail(), inner(), outer() })  // scan dimension is innermost
		.vectorize(inner(0), vector_width);                  // vectorize innermost non-scan dimension
	if (parallel)
		R.parallel(outer());                 // TODO: only parallelize outermost

	//if (max_tile > 0)
	//{
	//	R.compute_globally()
	//		.split(full(0), max_tile, inner(), outer())         // convert upto 3 full dimensions
	//		.split(full(0), max_tile, inner(), outer())         // into tiles
	//		.split(full(0), max_tile, inner(), outer())
	//		.reorder({ outer_scan(), tail(), inner(), outer() })  // scan dimension is innermost
	//		.vectorize(inner(0), vector_width)                  // vectorize innermost non-scan dimension
	//		.parallel(outer());                                 // TODO: only parallelize outermost
	//}
	//else
	//{
	//	R.compute_globally()
	//		.reorder({ outer_scan(), tail(), inner(), outer() })  // scan dimension is innermost
	//		.vectorize(inner(0), vector_width)                  // vectorize innermost non-scan dimension
	//		.parallel(outer());                                 // TODO: only parallelize outermost
	//}
}

// -----------------------------------------------------------------------------

void RecFilter::gpu_auto_schedule(int tile_width)
{
	if (contents.get()->tiled)
	{
		if (
			Algorithm == (Custom | SlidingSingle)
			|| Algorithm == (Custom | SlidingSinglePostProcess)
			|| Algorithm == (Custom | SlidingSingleMultiDelta)
			|| Algorithm == (Custom | SlidingSingleMultiDeltaPostProcess)
			)
		{
			if (slidingScheduled)
			{
				gpu_auto_schedule_for_sliding_custom();
			}
			else
			{
				gpu_auto_schedule_for_sliding();
			}
		}
		else
		{
			gpu_auto_intra_schedule(1);
			gpu_auto_intra_schedule(2);
			gpu_auto_inter_schedule();
		}
	}
	else
	{
		gpu_auto_full_schedule(tile_width);
	}
}

void RecFilter::gpu_auto_full_schedule(int tile_width)
{
	if (contents.get()->tiled)
	{
		cerr << "Filter is tiled, use RecFilter::gpu_auto_intra_schedule() "
			<< "and RecFilter::gpu_auto_inter_schedule()\n" << endl;
		assert(false);
	}

	int max_threads = RecFilter::max_threads_per_cuda_warp;
	if (max_threads <= 0)
	{
		cerr << "Use RecFilter::set_max_threads_per_cuda_warp() to specify the "
			<< "maximum number of threads in each CUDA warp" << endl;
		assert(false);
	}

	RecFilterSchedule R = full_schedule();

	R.compute_globally()
		.unroll(full_scan())
		.split(full(0), tile_width, inner(), outer())  // convert upto three full
		.split(full(0), tile_width, inner(), outer())  // dimensions into tiles
		.split(full(0), tile_width, inner(), outer());

	VarTag tx = inner(0);
	VarTag ty = inner(1);
	VarTag tz = inner(2);

	if (R.contains_vars_with_tag(ty) && tile_width * tile_width > max_threads)
	{
		int factor = tile_width * tile_width / max_threads;
		R.split(ty, factor).unroll(ty.split_var());
	}

	R.reorder({ full_scan(), ty.split_var(), tz, tx, ty, outer() })
		.gpu_threads(tx, ty)
		.gpu_blocks(outer(0), outer(1), outer(2));
}

void RecFilter::gpu_auto_inter_schedule(void)
{
	if (!contents.get()->tiled)
	{
		cerr << "Filter is not tiled, use RecFilter::gpu_auto_full_schedule()\n" << endl;
		assert(false);
	}

	if (contents.get()->filter_info.size() > AUTO_SCHEDULE_MAX_DIMENSIONS)
	{
		cerr << "Auto schedules are not supported for more than " << AUTO_SCHEDULE_MAX_DIMENSIONS << " filter dimensions" << endl;
		assert(false);
	}

	int max_threads = RecFilter::max_threads_per_cuda_warp;
	if (max_threads <= 0)
	{
		cerr << "Use RecFilter::set_max_threads_per_cuda_warp() to specify the "
			<< "maximum number of threads in each CUDA warp" << endl;
		assert(false);
	}

	RecFilterSchedule R = inter_schedule();

	if (R.empty())
	{
		return;
	}

	// max tile is the maximum number of threads that will be launched
	// by specifying either of tx, ty, or tz as parallel
	int max_tile = 0;
	for (int i = 0; i < contents.get()->filter_info.size(); i++)
	{
		if (contents.get()->filter_info[i].tile_width < contents.get()->filter_info[i].image_width)
		{
			max_tile = std::max(max_tile, contents.get()->filter_info[i].tile_width);
		}
	}

	// store inner dimensions innermost because threads are operating
	// on these dimensions - memory coalescing
	R.compute_globally()
		.reorder_storage({ full(), inner(), tail(), outer() });

	R.split(full(0), max_tile, inner(), outer())  // upto two full dimensions
		.split(full(0), max_tile, inner(), outer());

	VarTag sc = outer_scan();       // exactly one scan dimension

	VarTag tx = inner(0);           // at most 3 inner dimensions
	VarTag ty = inner(1);           // because 1 inner dimension is a tail

	VarTag bx = outer(0);           // at most 2 outer dimensions
	VarTag by = outer(1);           // as 1 outer dim is being scanned

	// split an outer dimensions to generate extra threads to fill CUDA warps;
	// any parallelism ty for 3D filters is not exploited
	int factor = max_threads / max_tile;

	R.unroll(sc)
		.split(bx, factor)
		.reorder({ sc, tail(), ty, bx.split_var(), tx, bx, by })
		.gpu_threads(tx, bx.split_var())
		.gpu_blocks(bx, by);
}

void RecFilter::gpu_auto_schedule_for_sliding(void)
{
	if (!contents.get()->tiled)
	{
		cerr << "Filter is not tiled, use RecFilter::cpu_auto_full_schedule()\n" << endl;
		assert(false);
	}

	RecFilterSchedule R = intra_schedule(0);

	if (R.empty())
	{
		return;
	}

	int max_tile = 0;
	for (int i = 0; i < contents.get()->filter_info.size(); i++)
	{
		if (contents.get()->filter_info[i].tile_width < contents.get()->filter_info[i].image_width)
		{
			max_tile = std::max(max_tile, contents.get()->filter_info[i].tile_width);
		}
	}

	// よくわからないがinfo[1]にデータがちゃんと入ってたらyフィルタ, info[0]ならxフィルタ
	bool isYfilter = contents.get()->filter_info[1].feedfwd_order > 0;

	if (isYfilter)
	{
		// 各フィルタのスケジュール
		// 1. フィルタ方向にsplit
		// 2. xi, yi, xo,yo, cでreorder
		// 3. xiをベクトル化
		// 
		// メモリ的に連続なx方向でベクトル化するのが最も速いためxiを最内部ループにしてベクトル化
		// yフィルタではunroll, paralellはしないほうが速い
		// split長=vectorize長, set_vectorization_witdhの値は使っていない
		R
			.split(full(0), max_tile, inner(), outer())
			.reorder({ inner(0),inner_scan(), outer(), outer_channels() })
			.vectorize(inner(0))
			.compute_locally(); // こいつを最後にしないと，内側のOUTERでcompute_atできない(元々一番上にあった)

		{
			// 内部のFuncを取り出す
			// func_listは各フィルタ
			// skinFuncは各フィルタの和を取るFuncでoutputそのもの
			vector<Func> func_list;
			RecFilterFunc skinFunc;

			map<string, RecFilterFunc>::iterator f_it = contents.get()->func.begin();
			for (; f_it != contents.get()->func.end(); f_it++)
			{
				if (f_it->second.func_type == SKIN)
				{
					skinFunc = f_it->second;
					continue;
				}

				FuncTag ftag = f_it->second.func_category;
				if (ftag == FuncTag(INTRA_1) | ftag == FuncTag(INTRA_N))
				{

					func_list.push_back(Func(f_it->second.func));
				}
			}

			if (func_list.empty())
			{
				cerr << "Warning: No intra tile functions to schedule" << endl;
			}

			// skinFuncのスケジューリング
			// 
			// フィルタ方向じゃないxでsplit, parallel, vectorize
			// フィルタ方向のyでparallelは遅くなるがxは速くなる
			// ↑でやった各フィルタのスケジューリングと同じreorder
			Var xo("xo"), xi("xi");
			Func(skinFunc.func)
				.split(Var("x"), xo, xi, 128)
				.reorder(xi, Var("yi"), xo, Var("yo"))
				//.parallel(Var("yo"))
				.parallel(Var("c"))
				//.parallel(xo)
				.vectorize(xi);

			// 各フィルタの追加スケジュール
			for (int k = 0; k < func_list.size(); k++)
			{
				// xi, yi, xo, yoの
				// yiでcompute_atすることで高速化 
				// yiでstoreすると、直近2要素が必要となるスライディング処理では不都合なのでxoでstore
				func_list[k]
					.compute_at(Func(skinFunc.func), Var("yi"))
					.store_at(Func(skinFunc.func), Var("xo"));

				// 各フィルタのループを1回のループに
				// xi, yi, xo, yoのxiをcompute_with
				// RecFilterではdefinitionはundef, update(0)でフィルタが定義されている
				if (k != func_list.size() - 1)
				{
					func_list[k].update(0).compute_with(func_list[k + 1].update(0), RVar("xi"));
				}
			}
		}
	}
	else
	{
		// 各フィルタのスケジュール
		// 1. フィルタ方向にsplit
		// 2. yi, xi, xo, yoでreorder
		// 3. yiをベクトル化
		// 
		// メモリ的に連続なx方向にベクトル化したいが、スライディング処理はフィルタ方向にベクトル化できないためyiでベクトル化
		// xフィルタでは無指定unrollすると速くなる
		// yフィルタではx parallelすると速くなる
		// split長=vectorize長, set_vectorization_witdhの値は使っていない
		R
			.split(full(0), max_tile, inner(), outer())
			.reorder({ inner(0),inner_scan(), outer(), outer_channels() })
			.vectorize(inner(0))
			.unroll(inner_scan())
			.compute_locally(); // こいつを最後にしないと，内側のOUTERでcompute_atできない(元々一番上にあった)

		//if (parallel)
			//R.parallel(outer());


		{
			// 内部のFuncを取り出す
			// func_listは各フィルタ
			// skinFuncは各フィルタの和を取るFuncでoutputそのもの
			vector<Func> func_list;
			RecFilterFunc skinFunc;

			map<string, RecFilterFunc>::iterator f_it = contents.get()->func.begin();
			for (; f_it != contents.get()->func.end(); f_it++)
			{
				if (f_it->second.func_type == SKIN)
				{
					skinFunc = f_it->second;
					continue;
				}

				FuncTag ftag = f_it->second.func_category;
				if (ftag == FuncTag(INTRA_1) | ftag == FuncTag(INTRA_N))
				{
					func_list.push_back(Func(f_it->second.func));
				}
			}

			if (func_list.empty())
			{
				cerr << "Warning: No intra tile functions to schedule" << endl;
			}

			Var outer;
			map<string, VarTag> pure_var_category = skinFunc.pure_var_category;
			map<string, VarTag>::iterator vit = pure_var_category.begin();
			map<string, VarTag>::iterator vend = pure_var_category.end();

			for (; vit != vend; vit++)
			{
				if (vit->second == VarTag(OUTER, 0))
				{
					outer = Var(vit->first);
				}
			}

			// skinFuncのスケジューリング
			// 
			// フィルタ方向じゃないyでsplit, parallel
			// vectorizeはy方向なのでしないほうが早い
			// reoderはしてもしなくても変わらない
			Func(skinFunc.func)
				.split(Var("y"), Var("yo"), Var("yi"), 128)
				.parallel(Var("c"))
				.parallel(Var("yo"));

			// 各フィルタの追加スケジュール
			for (int k = 0; k < func_list.size(); k++)
			{
				// yi, xi, xo, yoの
				// xoでcompute_atすることで高速化
				// 
				// yフィルタのようにstoreとcomputeを分けても速度は変わらなかった、恐らくベクトル化がyiでメモリ的に最適ではないのが原因
				func_list[k]
					.compute_at(Func(skinFunc.func), outer)
					.store_at(Func(skinFunc.func), outer);

				// 各フィルタのループを1回のループに
				// yi, xi, xo, yoのyiをcompute_with
				// RecFilterではdefinitionはundef, update(0)でフィルタが定義されている
				if (k != func_list.size() - 1)
				{
					func_list[k].update(0).compute_with(func_list[k + 1].update(0), RVar("yi"));
				}
			}
		}
	}
}

void RecFilter::gpu_auto_schedule_for_sliding_custom(void)
{
	if (!contents.get()->tiled)
	{
		cerr << "Filter is not tiled, use RecFilter::cpu_auto_full_schedule()\n" << endl;
		assert(false);
	}

	RecFilterSchedule R = intra_schedule(0);

	if (R.empty())
	{
		return;
	}

	int max_tile = 0;
	for (int i = 0; i < contents.get()->filter_info.size(); i++)
	{
		if (contents.get()->filter_info[i].tile_width < contents.get()->filter_info[i].image_width)
		{
			max_tile = std::max(max_tile, contents.get()->filter_info[i].tile_width);
		}
	}

	// よくわからないがinfo[1]にデータがちゃんと入ってたらyフィルタ, info[0]ならxフィルタ
	bool isYfilter = contents.get()->filter_info[1].feedfwd_order > 0;

	if (isYfilter)
	{
		// split
		//R.split(full(0), schedule.splitY, inner(), outer());

		// reorder
		vector<VarTag> varTag;
		for (ScheduleTag tag : schedule.reorder)
		{
			varTag.push_back(mapScheduleTagToVarTag(tag, true));
		}
		R.reorder(varTag);

		// vectorize
		for (std::pair<ScheduleTag, int> pair : schedule.vectorize)
		{
			R.vectorize(mapScheduleTagToVarTag(pair.first, true));
		}

		// parallelはなし

		// unroll
		for (std::pair<ScheduleTag, int> pair : schedule.unroll)
		{
			R.unroll(mapScheduleTagToVarTag(pair.first, true));
		}

		R
			.compute_locally();

		{
			// 内部のFuncを取り出す
			// func_listは各フィルタ
			// skinFuncは各フィルタの和を取るFuncでoutputそのもの
			vector<Func> func_list;
			RecFilterFunc skinFunc;

			map<string, RecFilterFunc>::iterator f_it = contents.get()->func.begin();
			for (; f_it != contents.get()->func.end(); f_it++)
			{
				if (f_it->second.func_type == SKIN)
				{
					skinFunc = f_it->second;
					continue;
				}

				FuncTag ftag = f_it->second.func_category;
				if (ftag == FuncTag(INTRA_1) | ftag == FuncTag(INTRA_N))
				{

					func_list.push_back(Func(f_it->second.func));
				}
			}

			if (func_list.empty())
			{
				cerr << "Warning: No intra tile functions to schedule" << endl;
			}

			// split
			Var xo("xo"), xi("xi");
			Func(skinFunc.func)
				.split(Var("x"), xo, xi, schedule.splitX);

			// gpu Block
			vector<VarOrRVar> gpuBlocksVar;
			for (ScheduleTag tag : schedule.gpuBlocks)
			{
				gpuBlocksVar.push_back(mapSchedulleTagToVar(tag));
			}
			switch (gpuBlocksVar.size())
			{
			case 1:
				Func(skinFunc.func)
					.gpu_blocks(gpuBlocksVar[0]);

				break;
			case 2:
				Func(skinFunc.func)
					.gpu_blocks(gpuBlocksVar[0], gpuBlocksVar[1]);

				break;
			case 3:
				Func(skinFunc.func)
					.gpu_blocks(gpuBlocksVar[0], gpuBlocksVar[1], gpuBlocksVar[2]);

				break;
			default:
				break;
			}

			// gpu Thread
			vector<VarOrRVar> gpuThreadsVar;
			for (ScheduleTag tag : schedule.gpuThreads)
			{
				gpuThreadsVar.push_back(mapSchedulleTagToVar(tag));
			}
			switch (gpuThreadsVar.size())
			{
			case 1:
				Func(skinFunc.func)
					.gpu_threads(gpuThreadsVar[0]);

				break;
			case 2:
				Func(skinFunc.func)
					.gpu_threads(gpuThreadsVar[0], gpuThreadsVar[1]);

				break;
			case 3:
				Func(skinFunc.func)
					.gpu_threads(gpuThreadsVar[0], gpuThreadsVar[1], gpuThreadsVar[2]);

				break;
			default:
				break;
			}

			// reorder
			vector<VarOrRVar> reorderVar;
			for (ScheduleTag tag : schedule.reorder)
			{
				reorderVar.push_back(mapSchedulleTagToVar(tag));
			}
			Func(skinFunc.func).reorder(reorderVar);

			 //vectorize
			for (std::pair<ScheduleTag, int> pair : schedule.vectorize)
			{
				if (pair.second == 0)
				{
					Func(skinFunc.func).vectorize(mapSchedulleTagToVar(pair.first));
				}
				else
				{
					Func(skinFunc.func).vectorize(mapSchedulleTagToVar(pair.first), pair.second);
				}
			}

			// unroll
			for (std::pair<ScheduleTag, int> pair : schedule.unroll)
			{
				if (pair.second == 0)
				{
					Func(skinFunc.func).unroll(mapSchedulleTagToVar(pair.first));
				}
				else
				{
					Func(skinFunc.func).unroll(mapSchedulleTagToVar(pair.first), pair.second);
				}
			}

			// 各フィルタの追加スケジュール
			// ここは変えない
			for (int k = 0; k < func_list.size(); k++)
			{
				// xi, yi, xo, yoの
				// yiでcompute_atすることで高速化 
				// yiでstoreすると、直近2要素が必要となるスライディング処理では不都合なのでxoでstore
				func_list[k]
					.compute_at(Func(skinFunc.func), Var("xi"));

					// gpuはparallelであり、store_atの下にparallelを作れない（多分）
					//.store_at(Func(skinFunc.func), Var("xo"));

				// 各フィルタのループを1回のループに
				// xi, yi, xo, yoのxiをcompute_with
				// RecFilterではdefinitionはundef, update(0)でフィルタが定義されている
				if (k != func_list.size() - 1)
				{
					func_list[k].update(0).compute_with(func_list[k + 1].update(0), RVar("ryi"));
				}
			}
		}
	}
	else
	{
		// split
		//R.split(full(0), schedule.splitX, inner(), outer());

		// reorder
		vector<VarTag> varTag;
		for (ScheduleTag tag : schedule.reorder)
		{
			varTag.push_back(mapScheduleTagToVarTag(tag, false));
		}
		R.reorder(varTag);

		// vectorize
		for (std::pair<ScheduleTag, int> pair : schedule.vectorize)
		{
			R.vectorize(mapScheduleTagToVarTag(pair.first, false));
		}

		// parallelはなし

		// unroll
		for (std::pair<ScheduleTag, int> pair : schedule.unroll)
		{
			R.unroll(mapScheduleTagToVarTag(pair.first, false));
		}

		R.compute_locally();

		{
			// 内部のFuncを取り出す
			// func_listは各フィルタ
			// skinFuncは各フィルタの和を取るFuncでoutputそのもの
			vector<Func> func_list;
			RecFilterFunc skinFunc;

			map<string, RecFilterFunc>::iterator f_it = contents.get()->func.begin();
			for (; f_it != contents.get()->func.end(); f_it++)
			{
				if (f_it->second.func_type == SKIN)
				{
					skinFunc = f_it->second;
					continue;
				}

				FuncTag ftag = f_it->second.func_category;
				if (ftag == FuncTag(INTRA_1) | ftag == FuncTag(INTRA_N))
				{
					func_list.push_back(Func(f_it->second.func));
				}
			}

			if (func_list.empty())
			{
				cerr << "Warning: No intra tile functions to schedule" << endl;
			}

			// split
			Func(skinFunc.func)
				.split(Var("y"), Var("yo"), Var("yi"), schedule.splitY)
				.reorder(Var("xi"), Var("yi"), Var("xo"), Var("yo"))
				;

			// xはskinとフィルタのスケジューリングが違うので難しい、暫定はなしで
			// gpu Block
			vector<VarOrRVar> gpuBlocksVar;
			for (ScheduleTag tag : schedule.gpuBlocks)
			{
				gpuBlocksVar.push_back(mapSchedulleTagToVar(tag));
			}
			switch (gpuBlocksVar.size())
			{
			case 1:
				Func(skinFunc.func)
					.gpu_blocks(gpuBlocksVar[0]);

				break;
			case 2:
				Func(skinFunc.func)
					.gpu_blocks(gpuBlocksVar[0], gpuBlocksVar[1]);

				break;
			case 3:
				Func(skinFunc.func)
					.gpu_blocks(gpuBlocksVar[0], gpuBlocksVar[1], gpuBlocksVar[2]);

				break;
			default:
				break;
			}

			// gpu Thread
			vector<VarOrRVar> gpuThreadsVar;
			for (ScheduleTag tag : schedule.gpuThreads)
			{
				gpuThreadsVar.push_back(mapSchedulleTagToVar(tag));
			}
			switch (gpuThreadsVar.size())
			{
			case 1:
				Func(skinFunc.func)
					.gpu_threads(gpuThreadsVar[0]);

				break;
			case 2:
				Func(skinFunc.func)
					.gpu_threads(gpuThreadsVar[0], gpuThreadsVar[1]);

				break;
			case 3:
				Func(skinFunc.func)
					.gpu_threads(gpuThreadsVar[0], gpuThreadsVar[1], gpuThreadsVar[2]);

				break;
			default:
				break;
			}

			// 各フィルタの追加スケジュール
			for (int k = 0; k < func_list.size(); k++)
			{
				// yi, xi, xo, yoの
				// xoでcompute_atすることで高速化
				// 
				// yフィルタのようにstoreとcomputeを分けても速度は変わらなかった、恐らくベクトル化がyiでメモリ的に最適ではないのが原因
				func_list[k]
					.compute_at(Func(skinFunc.func), Var("yi"))
					.store_at(Func(skinFunc.func), Var("yi"));

				// 各フィルタのループを1回のループに
				// yi, xi, xo, yoのyiをcompute_with
				// RecFilterではdefinitionはundef, update(0)でフィルタが定義されている
				if (k != func_list.size() - 1)
				{
					func_list[k].update(0).compute_with(func_list[k + 1].update(0), RVar("rxi"));
				}
			}
		}
	}
}

void RecFilter::gpu_auto_intra_schedule(int id)
{
	if (!contents.get()->tiled)
	{
		cerr << "Filter is not tiled, use RecFilter::gpu_auto_full_schedule()\n" << endl;
		assert(false);
	}

	if (contents.get()->filter_info.size() > AUTO_SCHEDULE_MAX_DIMENSIONS)
	{
		cerr << "Auto schedules are not supported for more than " << AUTO_SCHEDULE_MAX_DIMENSIONS << " filter dimensions" << endl;
		assert(false);
	}

	int max_threads = RecFilter::max_threads_per_cuda_warp;
	if (max_threads <= 0)
	{
		cerr << "Use RecFilter::set_max_threads_per_cuda_warp() to specify the "
			<< "maximum number of threads in each CUDA warp" << endl;
		assert(false);
	}

	RecFilterSchedule R = intra_schedule(id);

	if (R.empty())
	{
		return;
	}

	// max tile is the maximum number of threads that will be launched
	// by specifying either of tx, ty, or tz as parallel
	int max_tile = 0;
	int max_order = 0;
	int num_scans = 0;
	for (int i = 0; i < contents.get()->filter_info.size(); i++)
	{
		if (contents.get()->filter_info[i].tile_width < contents.get()->filter_info[i].image_width)
		{
			max_tile = std::max(max_tile, contents.get()->filter_info[i].tile_width);
		}
		max_order = std::max(max_order,
			std::max(contents.get()->filter_info[i].feedback_order, contents.get()->filter_info[i].feedfwd_order));
		num_scans += contents.get()->filter_info[i].num_scans;
	}

	R.split(full(0), max_tile, inner(), outer())  // upto two full dimensions
		.split(full(0), max_tile, inner(), outer());

	VarTag sc = inner_scan();       // exactly one scan dimension

	VarTag tx = inner(0);           // at most 3 inner dimensions
	VarTag ty = inner(1);
	VarTag tz = inner(2);

	VarTag bx = outer(0);           // at most 3 outer dimensions
	VarTag by = outer(1);
	VarTag bz = outer(2);

	R.compute_locally().storage_layout(INVALID, outer());

	switch (id)
	{
	case 1:
		// for nD intra tile terms: each inner var is of size max_tile if ty is not
		// empty then this will give too many threads, reduce by splitting ty; any
		// parallelism in tz in case of 3D filters is not exploited
		if (R.contains_vars_with_tag(ty) && max_tile * max_tile > max_threads)
		{
			int factor = max_tile * max_tile / max_threads;
			R.split(ty, factor).unroll(ty.split_var());
		}
		R.unroll(sc)
			.reorder({ sc, ty.split_var(), tz, tx, ty, outer() })
			.gpu_threads(tx, ty)
			.gpu_blocks(bx, by, bz);
		break;


	case 2:
		// for intra tile terms that compute cross dimensional residuals, generate more
		// threads by splitting an outer dimension and using it as threads; any parallelism
		// in tz in case of 3D filters is not exploited
		R.unroll(sc)
			.split(bx, max_tile / (num_scans * max_order))
			.reorder({ tx, ty, tz, sc, tail(), bx.split_var(), outer() })
			.fuse(tail(), tx)
			.gpu_threads(tail(), bx.split_var())
			.gpu_blocks(bx, by, bz);
		break;

	default: break;
	}
}

void RecFilter::schedule_for_sliding_custom(void)
{
	if (!schedule.gpuBlocks.empty() || !schedule.gpuThreads.empty())
	{
		RecFilter::set_max_threads_per_cuda_warp(128);
		gpu_auto_schedule_for_sliding_custom();
	}
	else
	{
		cpu_auto_schedule_for_sliding_custom();
	}
}

// -----------------------------------------------------------------------------

VarTag RecFilter::full(int i) { return VarTag(FULL, i); }
VarTag RecFilter::inner(int i) { return VarTag(INNER, i); }
VarTag RecFilter::outer(int i) { return VarTag(OUTER, i); }
VarTag RecFilter::tail(void) { return VarTag(TAIL); }
VarTag RecFilter::full_scan(void) { return VarTag(FULL | SCAN); }
VarTag RecFilter::inner_scan(void) { return VarTag(INNER | SCAN); }
VarTag RecFilter::outer_scan(void) { return VarTag(OUTER | SCAN); }
VarTag RecFilter::inner_channels(void) { return VarTag(INNER | CHANNEL); }
VarTag RecFilter::outer_channels(void) { return VarTag(OUTER | CHANNEL); }

// -----------------------------------------------------------------------------

Func RecFilter::as_func(void)
{
	if (contents.get()->func.empty())
	{
		cerr << "Filter " << contents.get()->name << " not defined" << endl;
		assert(false);
	}
	return Func(internal_function(contents.get()->name).func);
}

Func RecFilter::func(string func_name)
{
	map<string, RecFilterFunc>::iterator f = contents.get()->func.find(func_name);
	if (f != contents.get()->func.end())
	{
		return Func(f->second.func);
	}
	else
	{
		cerr << "Function " << func_name << " not found as a dependency of ";
		cerr << "recursive filter " << contents.get()->name << endl;
		assert(false);
	}
}

RecFilterFunc& RecFilter::internal_function(string func_name)
{
	map<string, RecFilterFunc>::iterator f = contents.get()->func.find(func_name);
	if (f != contents.get()->func.end())
	{
		return f->second;
	}
	else
	{
		cerr << "Function " << func_name << " not found as a dependency of ";
		cerr << "recursive filter " << contents.get()->name << endl;
		assert(false);
	}
}

std::vector<std::string> RecFilter::internal_functions(FuncTag ftag)
{
	std::vector<std::string> functions;
	std::map<std::string, RecFilterFunc>::iterator it = contents->func.begin();
	std::map<std::string, RecFilterFunc>::iterator end_it = contents->func.end();
	for (; it != end_it; it++)
	{
		if (it->second.func_category == ftag)
		{
			functions.push_back(it->first);
		}
	}
	return functions;
}

std::vector<std::string> RecFilter::internal_functions(FuncType ftype)
{
	std::vector<std::string> functions;
	std::map<std::string, RecFilterFunc>::iterator it = contents->func.begin();
	std::map<std::string, RecFilterFunc>::iterator end_it = contents->func.end();
	for (; it != end_it; it++)
	{
		if (it->second.func_type == ftype)
		{
			functions.push_back(it->first);
		}
	}
	return functions;
}

// -----------------------------------------------------------------------------

void RecFilter::compile_jit(string filename)
{
	if (!contents.get()->finalized)
	{
		finalize();
	}

	Func F = as_func();
	if (!filename.empty())
	{
		F.compile_to_lowered_stmt(filename, {}, HTML, contents.get()->target);
		//std::cout << "compile_jit : file name is empty!\n";
	}
	F.compile_jit(contents.get()->target);

	contents.get()->compiled = true;
}

Realization RecFilter::create_realization(void)
{
	// check if any of the functions have a schedule
	// true if all functions have default inline schedule
	bool no_schedule_applied = true;
	map<string, RecFilterFunc>::iterator fit;
	for (fit = contents.get()->func.begin(); fit != contents.get()->func.end(); fit++)
	{
		Function f = fit->second.func;
		f.lock_loop_levels();
		no_schedule_applied &= f.schedule().compute_level().is_inlined();
	}

	// apply a default schedule to compute everything
	// in global memory if no schedule has been used
	if (no_schedule_applied)
	{
		if (contents.get()->tiled)
		{
			inter_schedule().compute_globally();
			intra_schedule().compute_globally();
		}
		else
		{
			full_schedule().compute_globally();
		}
		cerr << "Warning: Applied default schedule to filter "
			<< contents.get()->name << endl;
	}

	// recompile the filter
	contents.get()->compiled = false;
	compile_jit();

	// upload all buffers to device if computed on GPU
	Func F(internal_function(contents.get()->name).func);
	if (contents.get()->target.has_gpu_feature())
	{
		map<string, Buffer<>> buff = extract_buffer_calls(F);
		for (map<string, Buffer<>>::iterator b = buff.begin(); b != buff.end(); b++)
		{
			b->second.copy_to_device();
			//b->second.copy_to_device();
		}
	}

	// allocate the buffer
	vector<int> buffer_size;
	for (int i = 0; i < contents.get()->filter_info.size(); i++)
	{
		buffer_size.push_back(contents.get()->filter_info[i].image_width);
	}

	// create a realization object
	vector<Buffer<>> buffers;
	for (int i = 0; i < F.outputs(); i++)
	{
		buffers.push_back(Buffer<>(contents.get()->type, buffer_size));
	}

	return Realization(buffers);
}

Realization RecFilter::realize(void)
{
	Func F(internal_function(contents.get()->name).func);
	Realization R = create_realization();
	F.realize(R, contents.get()->target);
	return R;
}

float RecFilter::profile(int iterations)
{
	Func F(internal_function(contents.get()->name).func);
	Realization R = create_realization();

	double total_time = 0;
	unsigned long time_start, time_end;

	if (contents.get()->target.has_gpu_feature())
	{
		F.realize(R, contents.get()->target); // warmup run

		time_start = millisecond_timer();
		for (int i = 0; i < iterations; i++)
		{
			F.realize(R, contents.get()->target);
		}
		time_end = millisecond_timer();
	}
	else
	{
		time_start = millisecond_timer();
		for (int i = 0; i < iterations; i++)
		{
			F.realize(R, contents.get()->target);
		}
		time_end = millisecond_timer();
	}
	total_time = (time_end - time_start);

	return total_time / iterations;
}

Target RecFilter::target(void)
{
	return contents.get()->target;
}

void RecFilter::setTargetFeature(Halide::Target::Feature feature)
{
	contents.get()->target.set_feature(feature);
}

// -----------------------------------------------------------------------------

string RecFilter::print_synopsis(void) const
{
	stringstream s;
	map<string, RecFilterFunc>::iterator f;
	for (f = contents.get()->func.begin(); f != contents.get()->func.end(); f++)
	{
		s << f->second << "\n";
	}
	s << "\n";
	return s.str();
}

string RecFilter::print_schedule(void) const
{
	stringstream s;
	map<string, RecFilterFunc>::iterator f;

	for (f = contents.get()->func.begin(); f != contents.get()->func.end(); f++)
	{
		map<int, vector<string> >::iterator sit;

		// dump the pure def schedule
		if (!f->second.pure_schedule.empty())
		{
			vector<string> str = f->second.pure_schedule;
			s << f->second.func.name();
			// first print any compute at rules
			bool compute_def_found = false;
			for (int i = 0; !compute_def_found && i < str.size(); i++)
			{
				if (str[i].find("compute_root") != string::npos ||
					str[i].find("compute_at") != string::npos)
				{
					s << "." << str[i];
					str.erase(str.begin() + i);
					compute_def_found = true;
				}
			}
			for (int i = 0; i < str.size(); i++)
			{
				if (str.size() < 2)
				{
					s << "." << str[i];
				}
				else
				{
					s << "\n    ." << str[i];
				}
			}
			s << ";\n";
		}

		// dump the update def schedules
		for (sit = f->second.update_schedule.begin(); sit != f->second.update_schedule.end(); sit++)
		{
			int def = sit->first;
			vector<string> str = sit->second;
			if (!str.empty())
			{
				s << f->second.func.name() << ".update(" << def << ")";
			}
			for (int i = 0; i < str.size(); i++)
			{
				s << "\n    ." << str[i];
			}
			s << ";\n";
		}
	}
	return s.str();
}

string RecFilter::print_functions(void) const
{
	stringstream s;
	map<string, RecFilterFunc>::iterator f;
	for (f = contents.get()->func.begin(); f != contents.get()->func.end(); f++)
	{
		s << f->second.func << "\n";
	}
	s << "\n";
	return s.str();
}

string RecFilter::print_hl_code(void) const
{
	string a = print_synopsis();
	string b = print_functions();
	string c = print_schedule();
	return a + b + c;
}

void RecFilter::inline_func(string func_name)
{
	if (contents.get()->name == func_name)
	{
		return;
	}

	// all the functions in this recfilter
	vector<Func> func_list;
	map<string, RecFilterFunc>::iterator f = contents.get()->func.begin();
	map<string, RecFilterFunc>::iterator fe = contents.get()->func.end();
	while (f != fe)
	{
		func_list.push_back(Func(f->second.func));
		f++;
	}

	// inline this function in all the functions of this filter
	Function F = internal_function(func_name).func;
	inline_function(F, func_list);
	contents.get()->func.erase(func_name);
}
