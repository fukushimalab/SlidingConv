#include "split.h"

#define CAUSAL					"Causal"
#define ANTICAUSAL				"AntiCausal"

struct CustomSplitInfo_SlidingSingle_PostProcess
{
	int feedfwd_order;                   ///< order of recursive filter in a given dimension
	int feedback_order;
	int filter_dim;                     ///< dimension id
	int num_scans;                      ///< number of scans in the dimension that must be tiled
	bool clamped_border;                ///< Image border expression (from RecFilterContents)

	int tile_width;                     ///< tile width for splitting
	int image_width;                    ///< image width in this dimension
	int num_tiles;                      ///< number of tile in this dimension

	int redundant;

	Halide::Type type;                  ///< filter output type
	Halide::Var  var;                   ///< variable that represents this dimension
	Halide::Var  inner_var;             ///< inner variable after splitting
	Halide::Var  outer_var;             ///< outer variable or tile index after splitting

	Halide::RDom rdom;                  ///< RDom update domain of each scan
	Halide::RDom inner_rdom;            ///< inner RDom of each scan
	Halide::RDom inner_rdom_redu;  ///< inner RDom for redundant
	Halide::RDom truncated_inner_rdom;  ///< inner RDom width a truncated
	Halide::RDom truncated_inner_rdom_redu;  ///< inner RDom width a truncated for redundant
	Halide::RDom outer_rdom;            ///< outer RDom of each scan
	Halide::RDom tail_rdom;             ///< RDom to extract the tail of each scan

	vector<bool> scan_causal;           ///< causal or anticausal flag for each scan
	vector<int>  scan_id;               ///< scan or update definition id of each scan

	vector<vector<Halide::Expr>> feedfwd_coeff_expr; ///< Feedforward coeffs (from RecFilterContents)
	vector<vector<Halide::Expr>> feedback_coeff_expr;///< Feedback coeffs  (from RecFilterContents)
};

/** Tiling info for each dimension of the filter */
static vector<CustomSplitInfo_SlidingSingle_PostProcess> recfilter_split_info;

/** All recursive filter funcs created during splitting transformations */
static map<string, RecFilterFunc> recfilter_func_list;

/// <summary>
/// splitし、再起処理を定義する
/// </summary>
/// <param name="rF"></param>
/// <param name="split_info"></param>
/// <returns></returns>
static RecFilterFunc create_final_tile_term(
	RecFilterFunc rF,
	vector<CustomSplitInfo_SlidingSingle_PostProcess> split_info)
{
	assert(!split_info.empty());

	Function F = rF.func;
	Function F_intra(F.name() + DASH + FINAL_TERM);

	// filter type
	Type type = split_info[0].type;

	// scheduling tags for function dimensions
	map<string, VarTag>          pure_var_category = rF.pure_var_category;
	vector<map<string, VarTag> > update_var_category = rF.update_var_category;

	// manipulate the pure def
	vector<string> pure_args = F.args();
	vector<Expr>   pure_values = F.values();

	for (int i = 0, o_cnt = 0, i_cnt = 0; i < split_info.size(); i++)
	{
		Var x = split_info[i].var;
		Var xi = split_info[i].inner_var;
		Var xo = split_info[i].outer_var;
		int tile_width = split_info[i].tile_width;

		// replace x by xi in LHS pure args
		// replace x by tile*xo+xi in RHS values
		for (int j = 0; j < pure_args.size(); j++)
		{
			if (pure_args[j] == x.name())
			{
				pure_args[j] = xi.name();
				pure_args.insert(pure_args.begin() + j + 1, xo.name());
				pure_var_category.erase(x.name());
				pure_var_category.insert(make_pair(xi.name(), VarTag(INNER, i_cnt++)));
				pure_var_category.insert(make_pair(xo.name(), VarTag(OUTER, o_cnt++)));
			}
		}
		for (int j = 0; j < pure_values.size(); j++)
		{
			pure_values[j] = substitute(x.name(), tile_width * xo + xi, pure_values[j]);
		}

		if (!F_intra.has_pure_definition())
			F_intra.define(pure_args, pure_values);
		else
		{
			Function _F_intra = create_function_same_group(F_intra);
			_F_intra.define(pure_args, pure_values);
			F_intra.definition() = _F_intra.definition();
		}
	}

	// split info object and split id for each scan
	vector< pair<int, int> > scan(F.updates().size());
	for (int i = 0; i < split_info.size(); i++)
	{
		for (int j = 0; j < split_info[i].num_scans; j++)
		{
			scan[split_info[i].scan_id[j]] = make_pair(i, j);
		}
	}

	// create the scans from the split info object
	vector<Definition> updates;
	for (int i = 0; i < scan.size(); i++)
	{
		CustomSplitInfo_SlidingSingle_PostProcess s = split_info[scan[i].first];

		Var x = s.var;
		Var xi = s.inner_var;
		Var xo = s.outer_var;
		RDom rx = s.rdom;
		RDom rxi = s.inner_rdom;
		int tile_width = s.tile_width;
		int num_tiles = s.num_tiles;
		int image_width = s.image_width;

		int filter_dim = s.filter_dim;
		//int filter_order = s.filter_order;
		int feedback_order = s.feedback_order;
		int feedfwd_order = s.feedback_order;
		bool causal = s.scan_causal[scan[i].second];
		bool clamped_border = s.clamped_border;
		int  dimension = -1;

		vector<Expr> feedfwd = s.feedfwd_coeff_expr[i];
		vector<Expr> feedback = s.feedback_coeff_expr[i];

		// number of inner and outer vars in the update def
		int i_cnt = 0;
		int o_cnt = 0;

		// update args: replace rx by the RVar of this dimension in rxi and xo
		// replace all other pure args with their respective RVar in rxi
		vector<Expr> args;
		for (int j = 0; j < F.args().size(); j++)
		{
			string a = F.args()[j];
			if (a == x.name())
			{
				RVar rvar = rxi[filter_dim];
				if (causal)
				{
					args.push_back(rvar);
				}
				else
				{
					args.push_back(tile_width - 1 - rvar);
				}
				dimension = args.size() - 1;
				args.push_back(xo);
				update_var_category[i].erase(rx.x.name());
				update_var_category[i].insert(make_pair(rvar.name(), INNER | SCAN));
				update_var_category[i].insert(make_pair(xo.name(), VarTag(OUTER, o_cnt++)));
			}
			else
			{
				bool found = false;
				for (int k = 0; !found && k < split_info.size(); k++)
				{
					if (a == split_info[k].var.name())
					{
						RVar rvar = rxi[split_info[k].filter_dim];
						Var  var = split_info[k].outer_var;
						args.push_back(rvar);
						args.push_back(var);

						update_var_category[i].erase(a);
						update_var_category[i].insert(make_pair(rvar.name(), VarTag(INNER, i_cnt++)));
						update_var_category[i].insert(make_pair(var.name(), VarTag(OUTER, o_cnt++)));

						found = true;
					}
				}
				if (!found)
				{
					args.push_back(Var(a));
				}
			}
		}
		assert(dimension >= 0);

		// update values: create the intra tile scans with special
		// borders for all tile on image boundary is clamped_border as specified
		// border for all internal tiles is zero
		vector<Expr> values(F_intra.outputs());
		for (int j = 0; j < values.size(); j++)
		{
			// これでOK?
			for (int k = 0; k < feedfwd.size(); k++)
			{
				if (k == 0)
				{
					values[j] = Cast::make(type, feedfwd[k]) * Call::make(F_intra, args, j);
					continue;
				}
				vector<Expr> call_args = args;
				if (causal)
				{
					call_args[dimension] = max(call_args[dimension] - k, 0);
				}
				else
				{
					call_args[dimension] = min(call_args[dimension] + k, tile_width - 1);
				}
				values[j] = Cast::make(type, feedfwd[k]) * Call::make(F_intra, call_args, j);
			}

			for (int k = 0; k < feedback.size(); k++)
			{
				vector<Expr> call_args = args;
				Expr first_tile = (causal ? (xo == 0) : (xo == num_tiles - 1));
				if (causal)
				{
					call_args[dimension] = max(call_args[dimension] - (k + 1), 0);
				}
				else
				{
					call_args[dimension] = min(call_args[dimension] + (k + 1), tile_width - 1);
				}

				// inner tiles must always be clamped to zero beyond tile borders
				// tiles on the image border unless clamping is specified in
				// which case only inner tiles are clamped to zero beyond ���E
				if (clamped_border)
				{
					values[j] += Cast::make(type, feedback[k]) *
						select(rxi[filter_dim] > k || first_tile,
							Call::make(F_intra, call_args, j), make_zero(type));
				}
				else
				{
					values[j] += Cast::make(type, feedback[k]) *
						select(rxi[filter_dim] > k,
							Call::make(F_intra, call_args, j), make_zero(type));
				}
			}
		}

		// 係数関係の処理
		// xが含まれていないか，Skin関数が含まれていないか
		for (int j = 0; j < values.size(); j++)
		{
			// 初めにSkin関係から変更
			values[j] = substitute_func_with_args_call(F.name(), x, F_intra, args, values[j]);

			Expr rep_x;
			if (causal)
			{
				rep_x = rxi[filter_dim] + xo * tile_width;
			}
			else
			{
				rep_x = (tile_width - 1 - rxi[filter_dim]) + xo * tile_width;
			}
			if (expr_uses_var(values[j], x.name()))
			{
				values[j] = substitute(x, rep_x, values[j]);
			}
		}


		F_intra.define_update(args, values);
	}

	RecFilterFunc rF_intra;
	rF_intra.func = F_intra;
	rF_intra.func_category = INTRA_N;
	rF_intra.func_type = FINAL;
	rF_intra.pure_var_category = pure_var_category;
	rF_intra.update_var_category = update_var_category;
	rF_intra.consumer_func = F.name();

	return rF_intra;
}

/// <summary>
///　add_filterをした数だけRecFilterFuncにフィルターをかけたfuncを生成する
/// </summary>
/// <param name="rF"></param>
/// <param name="split_info"></param>
/// <returns></returns>
static vector<RecFilterFunc> create_filter_funcs(
	RecFilterFunc& rF,
	vector<CustomSplitInfo_SlidingSingle_PostProcess> split_info)
{
	assert(!split_info.empty());

	string main_func_name = rF.consumer_func;

	Function F = rF.func;

	// filter type
	Type type = split_info[0].type;

	// scheduling tags for function dimensions
	map<string, VarTag>          pure_var_category = rF.pure_var_category;
	vector<map<string, VarTag> > update_var_category = rF.update_var_category;

	// manipulate the pure def
	vector<string> pure_args = F.args();
	vector<Expr>   pure_values = F.values();

	// 2回以上のsplitは想定しない
	if (split_info.size() >= 2)
	{
		assert(false);
	}

	CustomSplitInfo_SlidingSingle_PostProcess s = split_info[0];

	int tile = s.tile_width;
	int dim = s.filter_dim;
	int feedfwd_order = s.feedfwd_order;
	int feedback_order = s.feedback_order;
	Var x = s.var;
	Var xi = s.inner_var;
	Var xo = s.outer_var;
	RVar rxi = s.inner_rdom[s.filter_dim];

	int order = s.num_scans;

	vector<Function> F_cac(order);
	vector<RecFilterFunc> rf_cac(order);

	vector<Expr> undef_values(F.outputs(), undef(s.type));

	for (int k = 0; k < order; k++)
	{
		F_cac[k] = Function(F.name() + DASH + std::to_string(k));
		F_cac[k].define(pure_args, undef_values);
	}

	// 1個のスキャンがSliding-DCTのZk1個に対応する
	for (int j = 0; j < order; j++)
	{
		int  curr_scan = s.scan_id[j];
		vector<Expr> args;
		int xip = 0;
		for (int k = 0; k < pure_args.size(); k++)
		{
			if (pure_args[k] == xi.name())
			{
				if (s.scan_causal[j])
				{
					args.push_back(rxi);
				}
				else
				{
					args.push_back(tile - 1 - rxi);
				}
				xip = k;
			}
			else
			{
				args.push_back(Var(pure_args[k]));
			}
		}
		vector<Expr> call_args = args;
		vector<Expr> values;
		// 任意のオーダー実装本体
		for (int c = 0; c < F.outputs(); c++)
		{
			Expr in = pure_values[c];
			Expr value = 0.0f;

			for (int k = 0; k < feedfwd_order; k++)
			{
				Expr input;
				if (s.scan_causal[j])
				{
					input = substitute(xi.name(), rxi - k, in);
				}
				else
				{
					input = substitute(xi.name(), tile - 1 - rxi + k, in);
				}
				if (k == 0)
					value = s.feedfwd_coeff_expr[curr_scan][k] * input;
				else
					value += s.feedfwd_coeff_expr[curr_scan][k] * input;
			}

			for (int k = 0; k < feedback_order; k++)
			{
				if (s.scan_causal[j])
				{
					call_args[xip] = max(rxi - k - 1, 0);
				}
				else
				{
					call_args[xip] = min(tile - 1 - rxi + k + 1, tile - 1);
				}
				value += s.feedback_coeff_expr[curr_scan][k] * Call::make(F_cac[curr_scan], call_args, c);
			}
			values.push_back(value);
		}

		// 係数関係の処理
		// xが含まれていないか，Skin関数が含まれていないか
		for (int n = 0; n < values.size(); n++)
		{
			// 初めにSkin関係から変更
			values[n] = substitute_func_with_args_call(main_func_name, x, F_cac[curr_scan], args, values[n]);

			Expr rep_x;
			if (s.scan_causal[j])
			{
				rep_x = rxi + xo * tile;
			}
			else
			{
				rep_x = (tile - 1 - rxi) + xo * tile;
			}
			if (expr_uses_var(values[n], x.name()))
			{
				values[n] = substitute(x, rep_x, values[n]);
			}
		}

		F_cac[curr_scan].define_update(args, values);

		rf_cac[curr_scan].pure_var_category = pure_var_category;
		rf_cac[curr_scan].update_var_category.push_back(rF.pure_var_category);
		rf_cac[curr_scan].update_var_category[0].insert(make_pair(rxi.name(), INNER | SCAN));
		rf_cac[curr_scan].update_var_category[0].erase(xi.name());
		rf_cac[curr_scan].func_category = INTRA_1; // 元INTER INTRAでもOK?
		rf_cac[curr_scan].func_type = FINAL;
		rf_cac[curr_scan].func = F_cac[curr_scan];
	}

	return rf_cac;
}

/// <summary>
/// custom_init_funcを各funcのupdates(0)に入れる
/// </summary>
/// <param name="rF_cac"></param>
/// <param name="init_funcs"></param>
/// <param name="split_info"></param>
static void add_init_to_final_result_cac(
	vector<RecFilterFunc>& rF_cac,
	vector<Func> init_funcs,
	vector<CustomSplitInfo_SlidingSingle_PostProcess> split_info)
{
	// 2回以上のsplitは想定しない
	if (split_info.size() >= 2)
	{
		assert(false);
	}

	CustomSplitInfo_SlidingSingle_PostProcess sInfo = split_info[0];

	int order = sInfo.num_scans;

	// Sliding-DCTのZkの数だけ必要
	vector<Function> F_cac(order);
	vector<vector<string>> pure_args(order);
	vector<vector<Expr>>   pure_values(order);
	vector<vector<Definition>> updates_cac(order);
	vector<vector< map<string, VarTag>>> update_var_category(order);

	for (int i = 0; i < rF_cac.size(); i++)
	{
		F_cac[i] = rF_cac[i].func;
		pure_args[i] = F_cac[i].args();
		pure_values[i] = F_cac[i].values();
		updates_cac[i] = F_cac[i].updates();
		update_var_category[i] = rF_cac[i].update_var_category;
	}

	// new updates to be added for all the split updates
	vector<map<int, pair< vector<Expr>, vector<Expr>>>> new_updates_cac(order);

	vector<map<int, map<string, VarTag>>> new_update_var_category(order);

	vector<Function> init_functions;
	for (Func func : init_funcs)
	{
		init_functions.push_back(func.function());
	}

	int tile_width = sInfo.tile_width;
	int num_tiles = sInfo.num_tiles;
	int image_width = sInfo.image_width;
	RDom rxi = sInfo.inner_rdom;
	RDom rxt = sInfo.tail_rdom;
	RDom rxf = sInfo.truncated_inner_rdom;

	for (int s = 0; s < rF_cac.size(); s++)
	{
		for (int j = 0; j < updates_cac[s].size(); j++)
		{
			vector<Expr> args = updates_cac[s][j].args();
			new_update_var_category[s][j] = update_var_category[s][j];

			vector<Expr> values;
			vector<Expr> init_call_args = args;

			// 初期化Funcがスプリット状態ではないとき，引数をまとめる
			if (init_call_args.size() != init_functions[s].args().size())
			{
				init_call_args.clear();
				for (int arg_index = 0; arg_index < pure_args[s].size(); arg_index++)
				{
					if (pure_args[s][arg_index] == sInfo.inner_var.name())
					{
						Expr x = clamp(args[arg_index] + args[arg_index + 1] * tile_width, 0, image_width - 1);
						init_call_args.push_back(x);
						arg_index++;
					}
					else
					{
						init_call_args.push_back(args[arg_index]);
					}
				}
			}

			// undefの初期化関数は飛ばす
			bool flg = false;
			for (Expr v : init_functions[s].values())
			{
				if (Halide::Internal::is_undef(v))
				{
					flg = true;
				}
			}

			for (int k = 0; k < rF_cac[s].func.outputs(); k++)
			{
				if (flg)
				{
					values.push_back(Call::make(rF_cac[s].func, args, 0));
				}
				else
				{
					values.push_back(Call::make(init_functions[s], init_call_args, 0));
				}
			}


			// とりあえず冗長計算なし(おかしくなりそう)
			for (int k = 0; k < rxi.dimensions(); k++)
			{
				for (int u = 0; u < args.size(); u++)
				{
					args[u] = substitute(rxi[k].name(), rxt[k], args[u]);
				}
				for (int u = 0; u < values.size(); u++)
				{
					values[u] = substitute(rxi[k].name(), rxt[k], values[u]);
				}
				if (new_update_var_category[s][j].find(rxi[k].name()) != new_update_var_category[s][j].end())
				{
					VarTag v = new_update_var_category[s][j][rxi[k].name()];
					new_update_var_category[s][j].erase(rxi[k].name());
					new_update_var_category[s][j].insert(make_pair(rxt[k].name(), v));
				}

				// the new update runs the scan for the first t elements
				// change the reduction domain of the original update to
				// run from t onwards, t = filter order
				for (int u = 0; u < updates_cac[s][j].args().size(); u++)
				{
					updates_cac[s][j].args()[u] = substitute(rxi[k].name(), rxf[k], updates_cac[s][j].args()[u]);
				}
				for (int u = 0; u < updates_cac[s][j].values().size(); u++)
				{
					updates_cac[s][j].values()[u] = substitute(rxi[k].name(), rxf[k], updates_cac[s][j].values()[u]);
				}

				if (update_var_category[s][j].find(rxi[k].name()) != update_var_category[s][j].end())
				{
					VarTag vc = update_var_category[s][j][rxi[k].name()];
					update_var_category[s][j].erase(rxi[k].name());
					update_var_category[s][j].insert(make_pair(rxf[k].name(), vc));
				}
			}
			new_updates_cac[s][j] = make_pair(args, values);
		}
	}

	// add extra update steps
	vector<Function> _F_cac(order);

	vector<Expr> undefval;
	for (int i = 0; i < F_cac[0].outputs(); i++)
	{
		undefval.push_back(undef(F_cac[0].output_types()[i]));
	}

	for (int s = 0; s < F_cac.size(); s++)
	{
		rF_cac[s].update_var_category.clear();
		_F_cac[s] = Function(F_cac[s].name());

		_F_cac[s].define(pure_args[s], pure_values[s]); // updef?

		for (int i = 0; i < updates_cac[s].size(); i++)
		{
			if (new_updates_cac[s].find(i) != new_updates_cac[s].end())
			{
				vector<Expr> args = new_updates_cac[s][i].first;
				vector<Expr> values = new_updates_cac[s][i].second;

				_F_cac[s].define_update(args, values);

				rF_cac[s].update_var_category.push_back(new_update_var_category[s][i]);
			}
			_F_cac[s].define_update(updates_cac[s][i].args(), updates_cac[s][i].values());

			rF_cac[s].update_var_category.push_back(update_var_category[s][i]);
		}
		replace_function(F_cac[s], _F_cac[s]);
	}
	return;
}

/// <summary>
/// add_init_to_final_funcによって、updates[0]: init, updates[1]: filterとなって2回更新が走っているので
/// selectを用いて1回にまとめる
/// </summary>
/// <param name="rF_final"></param>
/// <param name="split_info"></param>
static void merge_init_and_filtering_term(vector<RecFilterFunc>& rF_final_calc, vector<CustomSplitInfo_SlidingSingle_PostProcess>& split_info)
{
	if (split_info.size() >= 2)
	{
		cerr << "Sliding cannot be split twice" << endl;
		assert(false);
	}

	CustomSplitInfo_SlidingSingle_PostProcess sInfo = split_info[0];

	const int filter_dim = sInfo.filter_dim;
	const int order = std::max(sInfo.feedfwd_order, sInfo.feedback_order);
	const int fiter_start_index = std::max(std::max(sInfo.feedfwd_order - 1, sInfo.feedback_order), 0);
	const int redu = sInfo.redundant;
	RDom rxf = (redu > 0) ? sInfo.truncated_inner_rdom_redu : sInfo.truncated_inner_rdom;
	RDom rxt = sInfo.tail_rdom;
	RDom rxtf = (redu > 0) ? sInfo.inner_rdom_redu : sInfo.inner_rdom;
	Var xi = sInfo.inner_var;

	for (int k = 0; k < rF_final_calc.size(); k++)
	{
		Function& F = rF_final_calc[k].func;
		Function _F(F.name());

		_F.define(F.args(), F.values());
		vector<string> pure_args = F.args();
		rF_final_calc[k].update_var_category.clear();

		int xi_pos = -1;
		for (int i = 0; i < pure_args.size(); i++)
		{
			if (pure_args[i] == xi.name())
				xi_pos = i;
		}

		for (int i = 0; i < F.updates().size(); i++)
		{
			vector<Expr> init_args = F.update(i).args();
			vector<Expr> init_values = F.update(i).values();

			i++; // 初期化のすぐ次にフィルタはあるはず
			vector<Expr> filter_args = F.update(i).args();
			vector<Expr> filter_values = F.update(i).values();

			vector<Expr> args = init_args;
			vector<Expr> values(F.outputs());

			args[xi_pos] = substitute(rxt[filter_dim], rxtf[filter_dim], args[xi_pos]);

			for (int j = 0; j < init_values.size(); j++)
			{
				values[j] = select(rxtf[filter_dim] < fiter_start_index, init_values[j], filter_values[j]);
				for (int n = 0; n < rxt.dimensions(); n++)
				{
					values[j] = substitute(rxt[n], min(rxtf[n], order - 1), values[j]);
					values[j] = substitute(rxf[n], rxtf[n], values[j]);
				}
			}

			_F.define_update(args, values);

			map<string, VarTag> update_var_category = rF_final_calc[k].pure_var_category;
			update_var_category.erase(xi.name());
			update_var_category.insert(make_pair(rxtf[filter_dim].name(), INNER | SCAN));
			rF_final_calc[k].update_var_category.push_back(update_var_category);
		}
		replace_function(F, _F);
	}
}

void RecFilter::split_custom_sliding_single_post_process(map<string, int> dim_tile)
{
	if (contents.get()->tiled)
	{
		cerr << "Recursive filter cannot be tiled twice" << endl;
		assert(false);
	}

	// clear global variables
	// TODO: remove the global vars and make them objects of the RecFilter class in some way
	contents.get()->finalized = false;
	contents.get()->compiled = false;
	recfilter_split_info.clear();
	recfilter_func_list.clear();

	// main function of the recursive filter that contains the final result
	RecFilterFunc& rF = internal_function(contents.get()->name);
	Function        F = rF.func;

	// group scans in same dimension together and change the order of splits accordingly
	contents.get()->filter_info = group_scans_by_dimension(F, contents.get()->filter_info);
	// inner RDom - has dimensionality equal to dimensions of the image
	// each dimension runs from 0 to tile width of the respective dimension
	vector<ReductionVariable> inner_scan_rvars;

	// inject tiling info into the FilterInfo structs
	for (int i = 0; i < contents.get()->filter_info.size(); i++)
	{
		if (dim_tile.find(contents.get()->filter_info[i].var.name()) != dim_tile.end())
		{

			// check that there are scans in this dimension
			if (contents.get()->filter_info[i].scan_id.empty())
			{
				cerr << "No scans to tile in dimension "
					<< contents.get()->filter_info[i].var.name() << endl;
				assert(false);
			}

			contents.get()->filter_info[i].tile_width = dim_tile[contents.get()->filter_info[i].var.name()];
		}

		Expr extent = 1;
		if (contents.get()->filter_info[i].tile_width != contents.get()->filter_info[i].image_width)
		{
			extent = contents.get()->filter_info[i].tile_width;
		}

		ReductionVariable r;
		r.min = 0;
		r.extent = extent;
		r.var = "r" + contents.get()->filter_info[i].var.name() + "i";
		inner_scan_rvars.push_back(r);
	}
	RDom inner_rdom = RDom(ReductionDomain(inner_scan_rvars));

	int num_scan = 0;

	// populate tile size and number of tiles for each dimension
	// populate the inner, outer and tail update domains to all dimensions
	for (map<string, int>::iterator it = dim_tile.begin(); it != dim_tile.end(); it++)
	{
		bool found = false;
		string x = it->first;
		int tile_width = it->second;
		for (int j = 0; !found && j < contents.get()->filter_info.size(); j++)
		{
			if (x != contents.get()->filter_info[j].var.name())
			{
				continue;
			}
			found = true;

			// no need to add a split if tile width is not same as image width
			assert(contents.get()->filter_info[j].tile_width == tile_width);
			if (contents.get()->filter_info[j].image_width == tile_width)
			{
				continue;
			}

			CustomSplitInfo_SlidingSingle_PostProcess s;

			// copy data from filter_info struct to split_info struct
			//s.filter_order = contents.get()->filter_info[j].filter_order;
			s.feedback_order = contents.get()->filter_info[j].feedback_order;
			s.feedfwd_order = contents.get()->filter_info[j].feedfwd_order;
			s.filter_dim = contents.get()->filter_info[j].filter_dim;
			s.num_scans = contents.get()->filter_info[j].num_scans;
			s.var = contents.get()->filter_info[j].var;
			s.rdom = contents.get()->filter_info[j].rdom;
			s.scan_causal = contents.get()->filter_info[j].scan_causal;
			s.scan_id = contents.get()->filter_info[j].scan_id;
			s.image_width = contents.get()->filter_info[j].image_width;
			s.tile_width = contents.get()->filter_info[j].tile_width;
			s.num_tiles = contents.get()->filter_info[j].image_width / tile_width;
			//s.tol = get_contents().get()->filter_info[j].tol; // set tol
			//s.max_iter = get_contents().get()->filter_info[j].max_iter; // set iter

			s.feedfwd_coeff_expr = contents.get()->feedfwd_coeff_expr;
			s.feedback_coeff_expr = contents.get()->feedback_coeff_expr;
			s.clamped_border = contents.get()->clamped_border;
			s.type = contents.get()->type;

			// set inner var and outer var
			s.inner_var = Var(x + "i");
			s.outer_var = Var(x + "o");

			// set inner rdom, same for all dimensions
			s.inner_rdom = inner_rdom;

			const int filter_order = std::max(s.feedfwd_order, s.feedfwd_order);

			// same as inner rdom except that the extent of scan dimension
			// is filter order rather than tile width
			vector<ReductionVariable> inner_tail_rvars = inner_scan_rvars;
			inner_tail_rvars[j].var = "r" + x + "t";
			inner_tail_rvars[j].min = 0;
			//inner_tail_rvars[j].extent = s.filter_order;
			inner_tail_rvars[j].extent = filter_order;
			s.tail_rdom = RDom(ReductionDomain(inner_tail_rvars));

			// same as inner rdom except that the domain is from filter_order to tile_width-1
			// instead of 0 to tile_width-1
			vector<ReductionVariable> inner_truncated_rvars = inner_scan_rvars;
			inner_truncated_rvars[j].var = "r" + x + "f";
			//inner_truncated_rvars[j].min = s.filter_order;
			inner_truncated_rvars[j].min = filter_order;
			//inner_truncated_rvars[j].extent = simplify(max(inner_truncated_rvars[j].extent - s.filter_order, 0));
			inner_truncated_rvars[j].extent = simplify(max(inner_truncated_rvars[j].extent - filter_order, 0));
			s.truncated_inner_rdom = RDom(ReductionDomain(inner_truncated_rvars));

			// 冗長計算用
			vector<ReductionVariable> inner_truncated_rvars_redu = inner_scan_rvars;
			inner_truncated_rvars_redu[j].var = "r" + x + "fx";
			//inner_truncated_rvars_redu[j].min = s.filter_order;
			inner_truncated_rvars_redu[j].min = filter_order;
			//inner_truncated_rvars_redu[j].extent = simplify(max(inner_truncated_rvars_redu[j].extent - (s.filter_order - this->redundant), 0));
			inner_truncated_rvars_redu[j].extent = simplify(max(inner_truncated_rvars_redu[j].extent - (filter_order - this->redundant), 0));
			s.truncated_inner_rdom_redu = RDom(ReductionDomain(inner_truncated_rvars_redu));

			vector<ReductionVariable> inner_rvars_redu = inner_scan_rvars;
			inner_rvars_redu[j].var = "r" + x + "tfx";
			inner_rvars_redu[j].min = 0;
			inner_rvars_redu[j].extent = simplify(max(inner_rvars_redu[j].extent + this->redundant, 0));
			s.inner_rdom_redu = RDom(ReductionDomain(inner_rvars_redu));

			s.redundant = this->redundant;


			// outer_rdom.x: over all tail elements of current tile
			// outer_rdom.y: over all tiles
			//s.outer_rdom = RDom(0, s.filter_order, 0, s.num_tiles, "r" + x + "o");
			s.outer_rdom = RDom(0, filter_order, 0, s.num_tiles, "r" + x + "o");

			recfilter_split_info.push_back(s);
			num_scan = std::max(s.num_scans, num_scan);
		}
		if (!found)
		{
			cerr << "Variable " << x << " does not correspond to any "
				<< "dimension of the recursive filter " << contents.get()->name << endl;
			assert(false);
		}
	}

	// return if there are no splits to apply
	if (recfilter_split_info.empty())
	{
		return;
	}


	// apply the actual splitting
	RecFilterFunc rF_final;
	vector<RecFilterFunc> rF_final_cac;
	{

		rF_final = create_final_tile_term(rF, recfilter_split_info);
		rF_final_cac = create_filter_funcs(rF_final, recfilter_split_info);

		for (int i = 0; i < rF_final_cac.size(); i++)
		{
			rF_final_cac[i].consumer_func = rF.func.name();
		}

		if (custom_init.size() != num_scan)
		{
			std::cerr << "The number of filters and initialization functions do not match\n";
			assert(false);
		}

		// add all the initialization to the final term
		add_init_to_final_result_cac(rF_final_cac, custom_init, recfilter_split_info);

		merge_init_and_filtering_term(rF_final_cac, recfilter_split_info);


		// add the intra, final and tail terms to the list functions
		for (RecFilterFunc rff : rF_final_cac)
		{
			recfilter_func_list.insert(make_pair(rff.func.name(), rff));
		}

		//for (auto a : rF_final_cac[0].func.definition().values())
		//{
		//	std::cout << a << endl;
		//}
		//for (auto a : rF_final_cac[0].func.updates())
		//{
		//	for (auto b : a.values())
		//	{
		//		std::cout << b << endl;
		//	}
		//}

		// k+1個のZkを足し合わせる最終的なFunction
		Function F_final = rF_final.func;

		vector<string> args = F_final.args();
		vector<Expr> values;
		vector<Expr> call_args;
		for (string a : args)
		{
			call_args.push_back(Var(a));
		}

		// k+1個のZkを足す
		for (int i = 0; i < F.outputs(); i++)
		{
			Expr val = Call::make(rF_final_cac[0].func, call_args, i);

			for (int k = 1; k < rF_final_cac.size(); k++)
			{
				val += Call::make(rF_final_cac[k].func, call_args, i);
			}
	
			values.push_back(val);
		}

		// Zkを足した物をpost_processに入れる
		// skin = post_process(values)
		// post_processもsplit
		if (has_post_process) {
			vector<Expr> pure_args = post_process.update_args();
			Expr post_process_value = post_process.update_value(0);

			Var x = recfilter_split_info[0].var;
			Var xi = recfilter_split_info[0].inner_var;
			Var xo = recfilter_split_info[0].outer_var;
			int tile_width = recfilter_split_info[0].tile_width;

			// post_processもSlidingとおなじsplitを
			post_process_value = substitute(x.name(), tile_width * xo + xi, post_process_value);

			// post_processには自分が含まれているので、そこをSlidingの結果に置き換える
			Expr post_process_call = Call::make(post_process.function(), pure_args);
			post_process_call = substitute(x.name(), tile_width * xo + xi, post_process_call);
			values[0] = substitute(post_process_call, values[0], post_process_value);
		}

		Function _F = create_function_same_group(F_final);
		_F.define(args, values);
		rF_final.func = _F;
		// remove the scheduling tags of the update defs
		rF_final.func_category = INLINE; // もとはINTRA_N INLINEでもOK?
		rF_final.func_type = FINAL;
		rF_final.update_var_category.clear();

		recfilter_func_list.insert(make_pair(rF_final.func.name(), rF_final));
	}
	// change the original function to index into the final term
	{
		Function F_final = rF_final.func;
		vector<string> args = F.args();
		vector<Expr> values;
		vector<Expr> call_args;
		for (int i = 0; i < F_final.args().size(); i++)
		{
			string arg = F_final.args()[i];
			call_args.push_back(Var(arg));
			for (int j = 0; j < recfilter_split_info.size(); j++)
			{
				Var var = recfilter_split_info[j].var;
				Var inner_var = recfilter_split_info[j].inner_var;
				Var outer_var = recfilter_split_info[j].outer_var;
				int tile_width = recfilter_split_info[j].tile_width;
				if (arg == inner_var.name())
				{
					call_args[i] = substitute(arg, var % tile_width, call_args[i]);
				}
				else if (arg == outer_var.name())
				{
					call_args[i] = substitute(arg, var / tile_width, call_args[i]);
				}
			}
		}
		for (int i = 0; i < F.outputs(); i++)
		{
			Expr val = Call::make(F_final, call_args, i);
			values.push_back(val);
		}
		Function _F = create_function_same_group(F);
		_F.define(args, values);
		rF.func = _F; //必須

		// remove the scheduling tags of the update defs
		rF.func_category = REINDEX;
		rF.func_type = SKIN;
		rF.producer_func = rF_final_cac[0].func.name();
		rF.update_var_category.clear();

		// split the tiled vars of the final term
		for (int i = 0; i < recfilter_split_info.size(); i++)
		{
			Var var = recfilter_split_info[i].var;
			Var inner_var = recfilter_split_info[i].inner_var;
			Var outer_var = recfilter_split_info[i].outer_var;
			int tile_width = recfilter_split_info[i].tile_width;

			Func(rF.func).split(var, outer_var, inner_var, tile_width);
			string s = "split(Var(\"" + var.name() + "\"), Var(\"" + outer_var.name()
				+ "\"), Var(\"" + inner_var.name() + "\"), " + std::to_string(tile_width) + ")";
			rF.pure_var_category.erase(var.name());
			rF.pure_var_category.insert(make_pair(inner_var.name(), VarTag(INNER, i)));
			rF.pure_var_category.insert(make_pair(outer_var.name(), VarTag(OUTER, i)));
			rF.pure_var_splits.insert(make_pair(outer_var.name(), var.name()));
			rF.pure_var_splits.insert(make_pair(inner_var.name(), var.name()));
			rF.pure_schedule.push_back(s);
		}
		recfilter_func_list.insert(make_pair(rF.func.name(), rF));
	}

	// add all the generated RecFilterFuncs
	contents.get()->func.insert(recfilter_func_list.begin(), recfilter_func_list.end());

	contents.get()->tiled = true;

	// perform generic and target dependent optimizations
	finalize();

	recfilter_func_list.clear();
	recfilter_split_info.clear();
}