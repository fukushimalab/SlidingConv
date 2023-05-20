#pragma once
#include "recfilter.h"
#include "recfilter_internals.h"
#include "coefficients.h"
#include "modifiers.h"

using namespace Halide;
using namespace Halide::Internal;

using std::string;
using std::cerr;
using std::endl;
using std::vector;
using std::map;
using std::pair;
using std::make_pair;
using std::swap;

/** String constants used to construct names of intermediate functions generated during tiling */
// {@
#define INTRA_TILE_RESULT      "Intra"
#define INTRA_TILE_TAIL_TERM   "Tail"
#define INTER_TILE_TAIL_SUM    "CTail"
#define COMPLETE_TAIL_RESIDUAL "TDeps"
#define FINAL_RESULT_RESIDUAL  "Deps"
#define FINAL_TERM             "Final"
#define SUB                    "Sub"
#define DASH                   '_'
// @}


/** Info required to split a particular dimension of the recursive filter */
struct SplitInfo
{
	int feedback_order;                   ///< order of recursive filter in a given dimension
	int feedfwd_order;
	int filter_dim;                     ///< dimension id
	int num_scans;                      ///< number of scans in the dimension that must be tiled
	bool clamped_border;                ///< Image border expression (from RecFilterContents)

	int tile_width;                     ///< tile width for splitting
	int image_width;                    ///< image width in this dimension
	int num_tiles;                      ///< number of tile in this dimension

	int redudant;

	Halide::Type type;                  ///< filter output type
	Halide::Var  var;                   ///< variable that represents this dimension
	Halide::Var  inner_var;             ///< inner variable after splitting
	Halide::Var  outer_var;             ///< outer variable or tile index after splitting

	Halide::RDom rdom;                  ///< RDom update domain of each scan
	Halide::RDom inner_rdom;            ///< inner RDom of each scan
	Halide::RDom truncated_inner_rdom;  ///< inner RDom width a truncated
	Halide::RDom truncated_inner_rdom_redu;  ///< inner RDom width a truncated for redundant
	Halide::RDom inner_rdom_redu;  ///< inner RDom for redundant
	Halide::RDom outer_rdom;            ///< outer RDom of each scan
	Halide::RDom tail_rdom;             ///< RDom to extract the tail of each scan

	vector<bool> scan_causal;           ///< causal or anticausal flag for each scan
	vector<int>  scan_id;               ///< scan or update definition id of each scan

	Halide::Buffer<float> feedfwd_coeff; ///< Feedforward coeffs (from RecFilterContents)
	Halide::Buffer<float> feedback_coeff;///< Feedback coeffs  (from RecFilterContents)
};



/**
 * Reorder the update defs such that update defs in first dimension come first,
 * followed by next dimension and so on; this can be performed because dimensions
 * are separable and this allows clean tiling semantics
 *
 * \param[in] F function containing scans in multiple dimensions
 * \param[in] filter_info scan info aboout all dimensions
*/
static vector<FilterInfo> group_scans_by_dimension(Function F, vector<FilterInfo> filter_info)
{
	vector<string> args = F.args();
	vector<Expr>  values = F.values();
	vector<Definition> updates = F.updates();

	vector<Definition> new_updates;
	vector<FilterInfo>       new_filter_info = filter_info;

	// use all scans with dimension 0 first, then 1 and so on
	for (int i = 0; i < filter_info.size(); i++)
	{
		for (int j = 0; j < filter_info[i].num_scans; j++)
		{
			int curr = filter_info[i].num_scans - 1 - j;
			int scan = filter_info[i].scan_id[curr];
			new_updates.push_back(updates[scan]);
			new_filter_info[i].scan_id[curr] = new_updates.size() - 1;
		}
	}
	assert(new_updates.size() == updates.size());

	// reorder the update definitions as per the new order
	Function _F = create_function_same_group(F);
	_F.define(args, values);
	for (int i = 0; i < new_updates.size(); i++)
	{
		_F.define_update(new_updates[i].args(), new_updates[i].values());
	}
	// _F -> F にコピー
	replace_function(F, _F);
	return new_filter_info;
}


/** Convert the pure def into the first update def and leave the pure def undefined
 * \param[in,out] rF function to be modified
 * \param[in] split_info tiling metadata
 */
static void convert_pure_def_into_first_update_def( // koko
	RecFilterFunc& rF,
	vector<SplitInfo> split_info)
{
	assert(!split_info.empty());

	Function F = rF.func;

	// nothing needed if the function is pure
	if (F.is_pure())
	{
		return;
	}
	vector<string> pure_args = F.args();
	vector<Expr> values = F.values();
	vector<Definition> updates = F.updates();

	// scheduling tags for the new update def are same as pure def
	map<string, VarTag> update_var_category = rF.pure_var_category;

	// leave the pure def undefined
	vector<Expr> undef_values(F.outputs(), undef(split_info[0].type));
	Function _F = create_function_same_group(F);
	_F.define(pure_args, undef_values);

	// initialize the buffer in the first update def
	// replace the scheduling tags of xi by rxi
	{
		vector<Expr> args;
		for (int j = 0; j < pure_args.size(); j++)
		{
			args.push_back(Var(pure_args[j]));
		}
		for (int i = 0; i < split_info.size(); i++)
		{
			Var  xo = split_info[i].outer_var;
			Var  xi = split_info[i].inner_var;
			RVar rxi = split_info[i].inner_rdom[split_info[i].filter_dim];
			for (int j = 0; j < args.size(); j++)
			{
				args[j] = substitute(xi.name(), rxi, args[j]);
			}
			for (int j = 0; j < values.size(); j++)
			{
				values[j] = substitute(xi.name(), rxi, values[j]);
			}
			VarTag vc = update_var_category[xi.name()];
			update_var_category.erase(xi.name());
			update_var_category.insert(make_pair(rxi.name(), vc));
		}
		_F.define_update(args, values);
	}

	// add all the other scans
	for (int i = 0; i < updates.size(); i++)
	{
		_F.define_update(updates[i].args(), updates[i].values());
	}

	rF.func = _F; // 必須

	// add the scheduling tags for the new update def in front of tags for
	// all other update defsstd::cout << "update_var_category...\n";
	rF.update_var_category.insert(rF.update_var_category.begin(), update_var_category);
	//前の処理でrxiをrxtに置換しているが、こいつがそれを元に戻すから、argとupdate_varで対応が崩れる
}



static RecFilterFunc create_copy(RecFilterFunc rF, string func_name)
{
	Function F = rF.func;
	Function B(func_name);

	// same pure definition
	B.define(F.args(), F.values());

	// replace all calls to old function with the copy
	for (int i = 0; i < F.updates().size(); i++)
	{
		Definition r = F.updates()[i];
		vector<Expr> args;
		vector<Expr> values;
		for (int j = 0; j < r.args().size(); j++)
		{
			args.push_back(substitute_func_call(F.name(), B, r.args()[j]));
		}
		for (int j = 0; j < r.values().size(); j++)
		{
			values.push_back(substitute_func_call(F.name(), B, r.values()[j])); // r.value()[j]が原因？？？
		}
		B.define_update(args, values);
	}

	// copy the scheduling tags
	RecFilterFunc rB;
	rB.func = B;
	rB.func_category = rF.func_category;
	rB.pure_var_category = rF.pure_var_category;
	rB.update_var_category = rF.update_var_category;

	return rB;
}