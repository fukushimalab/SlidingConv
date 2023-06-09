﻿#include "recfilter.h"
#include "recfilter_internals.h"
#include "modifiers.h"

using std::cerr;
using std::endl;
using std::vector;
using std::map;
using std::string;
using std::stringstream;
using std::pair;
using std::make_pair;

using namespace Halide;
using namespace Halide::Internal;

#define PURE_DEF  (-1)

static const VarOrRVar GPU_THREAD[] =
{
	VarOrRVar("__thread_id_x",false),
	VarOrRVar("__thread_id_y",false),
	VarOrRVar("__thread_id_z",false),
};

static const VarOrRVar GPU_BLOCK[] =
{
	VarOrRVar("__block_id_x",false),
	VarOrRVar("__block_id_y",false),
	VarOrRVar("__block_id_z",false),
};

// -----------------------------------------------------------------------------

static string remove_dollar(string s)
{
	string r = s;
	int pos = std::min(r.find("$"), r.find("_"));
	if (pos != r.npos)
	{
		r.erase(pos);
	}
	return (r.empty() ? "tmp" : r);
}

// -----------------------------------------------------------------------------

/** Remove the pure def of a Function and arithmetically add it to
 * the expression of the first update def; leaving the pure def undefined */
static void add_pure_def_to_first_update_def(Function f)
{
	vector<string> args = f.args();
	vector<Expr>   values = f.values();
	vector<Definition> updates = f.updates();

	// nothing to do if function has no update defs
	// or if all pure defs are undefined
	bool pure_def_undefined = is_undef(values);
	if (pure_def_undefined || updates.empty())
	{
		return;
	}

	// add pure def to the first update def
	for (int j = 0; j < updates[0].values().size(); j++)
	{
		// replace pure args by update def args in the pure value
		Expr val = values[j];
		for (int k = 0; k < args.size(); k++)
		{
			val = substitute(args[k], updates[0].args()[k], val);
		}

		// remove let statements in the expression because we need to
		// compare calling args
		updates[0].values()[j] = remove_lets(updates[0].values()[j]);

		// remove call to current pixel of the function
		updates[0].values()[j] = substitute_func_call_with_args(f.name(),
			updates[0].args(), val, updates[0].values()[j]);
	}

	// set all pure defs to zero or undef
	for (int i = 0; i < values.size(); i++)
	{
		values[i] = undef(f.output_types()[i]);
	}

	//f.definition();
	Function _f(f.name());
	_f.define(args, values);
	for (int i = 0; i < updates.size(); i++)
	{
		_f.define_update(updates[i].args(), updates[i].values());
	}
	replace_function(f, _f);
}

// -----------------------------------------------------------------------------

RecFilterSchedule::RecFilterSchedule(RecFilter& r, vector<string> fl) :
	recfilter(r), func_list(fl) {}


bool RecFilterSchedule::empty(void)
{
	return func_list.empty();
}

bool RecFilterSchedule::contains_vars_with_tag(VarTag vtag)
{
	bool empty = true;
	for (int j = 0; j < func_list.size(); j++)
	{
		RecFilterFunc f = recfilter.internal_function(func_list[j]);
		map< int, vector<VarOrRVar> > var_list = var_list_by_tag(f, vtag);

		Func F(f.func);

		// check if valid pure definition has any vars with this tag
		if (!is_undef(F.values()) && var_list.find(PURE_DEF) != var_list.end())
		{
			empty &= var_list[PURE_DEF].empty();
		}

		// check if any valid update definition has any vars with this tag
		for (int i = 0; i < F.num_update_definitions(); i++)
		{
			if (!is_undef(F.update_values(i)) && var_list.find(i) != var_list.end())
			{
				empty &= var_list[i].empty();
			}
		}
	}
	return !empty;
}

map< int, vector<VarOrRVar> > RecFilterSchedule::var_list_by_tag(RecFilterFunc f, VarTag vtag)
{
	bool ignore_count = !vtag.has_count();
	map< int, vector<VarOrRVar> > var_list;
	map<string, VarTag>::iterator vit;

	for (vit = f.pure_var_category.begin(); vit != f.pure_var_category.end(); vit++)
	{
		if (ignore_count)
		{
			if (vit->second.same_except_count(vtag))
			{
				var_list[PURE_DEF].push_back(Var(vit->first));
			}
		}
		else
		{
			if (vit->second == vtag)
			{
				var_list[PURE_DEF].push_back(Var(vit->first));
			}
		}
	}
	for (int i = 0; i < f.update_var_category.size(); i++)
	{
		for (vit = f.update_var_category[i].begin(); vit != f.update_var_category[i].end(); vit++)
		{
			if (ignore_count)
			{
				if (vit->second.same_except_count(vtag))
				{
					var_list[i].push_back(Var(vit->first));
				}
			}
			else
			{
				if (vit->second == vtag)
				{
					var_list[i].push_back(Var(vit->first));
				}
			}
		}
	}

	return var_list;
}

map<int, VarOrRVar> RecFilterSchedule::var_by_tag(RecFilterFunc f, VarTag vtag)
{
	map<int, VarOrRVar> var_list;
	map<string, VarTag>::iterator vit;
	for (vit = f.pure_var_category.begin(); vit != f.pure_var_category.end(); vit++)
	{
		if (vit->second == vtag)
		{
			if (var_list.find(PURE_DEF) == var_list.end())
			{
				var_list.insert(make_pair(PURE_DEF, Var(vit->first)));
			}
			else
			{
				cerr << "Found multiple vars with the scheduling tag " << vtag
					<< " in pure def of " << f.func.name() << endl;
				cerr << f << endl;
				assert(false);
			}
		}
	}
	for (int i = 0; i < f.update_var_category.size(); i++)
	{
		for (vit = f.update_var_category[i].begin(); vit != f.update_var_category[i].end(); vit++)
		{
			if (vit->second == vtag)
			{
				if (var_list.find(i) == var_list.end())
				{
					var_list.insert(make_pair(i, Var(vit->first)));
				}
				else
				{
					cerr << "Found multiple vars with the scheduling tag " << vtag
						<< " in update def " << i << " of " << f.func.name() << endl;
					cerr << f << endl;
					assert(false);
				}
			}
		}
	}

	return var_list;
}

// -----------------------------------------------------------------------------

RecFilterSchedule& RecFilterSchedule::compute_globally(void)
{
	for (int j = 0; j < func_list.size(); j++)
	{
		RecFilterFunc& rF = recfilter.internal_function(func_list[j]);
		Func            F = Func(rF.func);

		F.function().lock_loop_levels();
		// nothing to do if the function is already compute root
		if (F.function().schedule().compute_level().is_root() &&
			F.function().schedule().store_level().is_root())
		{
			continue;
		}

		// remove the initializations of all inter tile scans to avoid extra
		// kernel execution for initializing the output buffer for GPU schedule
		// this also ensure that this module is only enabled if the filter is tiled
		if (rF.func_category == INTER)
		{
			add_pure_def_to_first_update_def(F.function());
		}
		F.function().lock_loop_levels();
		if (F.function().schedule().compute_level().is_inlined() &&
			F.function().schedule().store_level().is_inlined())
		{
			F.compute_root();
			rF.pure_schedule.push_back("compute_root()");
		}
		else
		{
			cerr << "Cannot set " << F.name() << " to be computed in global memory "
				<< "because it is already scheduled to be computed locally" << endl;
			assert(false);
		}
	}
	return *this;
}

RecFilterSchedule& RecFilterSchedule::compute_locally(void)
{
	for (int j = 0; j < func_list.size(); j++)
	{
		RecFilterFunc& rF = recfilter.internal_function(func_list[j]);
		Func            F = Func(rF.func);

		// ignore if this is a reindexing function
		if (rF.func_category == REINDEX)
		{
			continue;
		}
		if (rF.external_consumer_func.defined())
			continue;


		// function and loop level within which this function must be computed
		Func consumer_func;
		Var  consumer_var;

		// go to all internal functions and find one whose producer is this function
		// there can be at most 1 such function
		bool found = false;
		for (int i = 0; i < func_list.size() && !found; i++)
		{
			RecFilterFunc& rf = recfilter.internal_function(func_list[i]);
			if (rf.func_category == REINDEX && rf.producer_func == F.name())
			{
				found = true;
				// if this has an external consumer then just
				// use the external consumer
				if (rf.external_consumer_func.defined())
				{
					consumer_func = rf.external_consumer_func;
					consumer_var = rf.external_consumer_var;
				}
				else
				{
					consumer_func = Func(rf.func);
					consumer_func.compute_root();

					rf.pure_schedule.push_back("compute_root()");

					// find the innermost tile index to use as loop level
					// GPU targets - this is CUDA tile index x
					// CPU targets - this is first OUTER index
					if (recfilter.target().has_gpu_feature())
					{
						//consumer_var = consumer_var.gpu_blocks();
						std::cout << "hoge\n";
					}
					else {
						map<int, VarOrRVar> outer_vlist = var_by_tag(rf, VarTag(OUTER, 0));
						if (outer_vlist.find(PURE_DEF) == outer_vlist.end())
						{
							cerr << F.name() << " cannot be computed locally in another function "
								<< "because the function calling it does not seem to have outer "
								<< "variable where " << F.name() << " can be computed" << endl;
							assert(false);
						}
						consumer_var = outer_vlist.find(PURE_DEF)->second.var;
					}
				}
			}
		}

		// external_consumerが優先
		/*if (!consumer_func.defined() && rF.external_consumer_func.defined())
		{
			consumer_func = rF.external_consumer_func;
			consumer_var = rF.external_consumer_var;
		}*/

		// consumer関数が定義されてたら問答無用でcomput_atにもっていく
		if (!consumer_func.defined() && recfilter.internal_function(rF.consumer_func).func.has_pure_definition())
		{
			consumer_func = Func(recfilter.internal_function(rF.consumer_func).func);
			map<int, VarOrRVar> outer_vlist = var_by_tag(rF, VarTag(OUTER, 0));
			consumer_var = outer_vlist.find(PURE_DEF)->second.var;
		}

		if (consumer_func.defined())
		{
			string compute_level_str = "compute_at(" + consumer_func.name() + ", Var(\""
				+ consumer_var.name() + "\"))";

			// functions consumed by this function should be computed at same level
			for (int i = 0; i < func_list.size(); i++)
			{
				RecFilterFunc& rf = recfilter.internal_function(func_list[i]);
				if (rf.func_category == REINDEX && rf.consumer_func == F.name())
				{
					Func(rf.func).compute_at(consumer_func, consumer_var);
					rf.pure_schedule.push_back(compute_level_str);
				}
			}
			F.compute_at(consumer_func, consumer_var);
			rF.pure_schedule.push_back(compute_level_str);

		}
		else
		{
			cerr << "Warning: " << F.name() << " cannot be computed locally in "
				<< "another function because it is not consumed by any function" << endl;
			compute_globally();
		}
	}
	return *this;
}

RecFilterSchedule& RecFilterSchedule::parallel(VarTag vtag, int factor)
{
	if (recfilter.target().has_gpu_feature())
	{
		cerr << "Cannot use RecFilterSchedule::parallel() if compilation "
			<< "target is GPU; use RecFilterSchedule::gpu_blocks() or "
			<< "RecFilterSchedule::gpu_threads()" << endl;
		assert(false);
	}

	if (vtag.check(SCAN))
	{
		cerr << "Cannot create parallel threads from scan variable" << endl;
		assert(false);
	}

	for (int j = 0; j < func_list.size(); j++)
	{
		RecFilterFunc& rF = recfilter.internal_function(func_list[j]);
		Func            F = Func(rF.func);
		// inline済みの皮になる関数はスケジュールなし
		if (rF.func_category == REINDEX && rF.external_consumer_func.defined())
		{
			continue;
		}
		map<int, vector<VarOrRVar> > vars = var_list_by_tag(rF, vtag);
		map<int, vector<VarOrRVar> >::iterator vit;

		for (vit = vars.begin(); vit != vars.end(); vit++)
		{
			int def = vit->first;
			vector<VarOrRVar> vlist = vit->second;

			for (int i = 0; i < vlist.size(); i++)
			{
				VarOrRVar v = vlist[i];

				// only add scheduling to defs that are not undef
				if (def == PURE_DEF)
				{
					if (!is_undef(F.values()))
					{
						if (factor)
						{
							F.parallel(v, factor);
							rF.pure_schedule.push_back("parallel(Var(\"" + v.name() + "\")," + std::to_string(factor) + ")");
						}
						else
						{
							F.parallel(v);
							rF.pure_schedule.push_back("parallel(Var(\"" + v.name() + "\"))");
						}
					}
				}
				else
				{
					if (!is_undef(F.update_values(def)))
					{
						if (factor)
						{
							F.update(def).parallel(v, factor);
							rF.update_schedule[def].push_back("parallel(Var(\"" + v.name() + "\")," + std::to_string(factor) + ")");
						}
						else
						{
							F.update(def).parallel(v);
							rF.update_schedule[def].push_back("parallel(Var(\"" + v.name() + "\"))");
						}
					}
				}
			}
		}
	}
	return *this;
}

RecFilterSchedule& RecFilterSchedule::unroll(VarTag vtag, int factor)
{
	for (int j = 0; j < func_list.size(); j++)
	{
		RecFilterFunc& rF = recfilter.internal_function(func_list[j]);
		Func            F = Func(rF.func);

		map<int, vector<VarOrRVar> > vars = var_list_by_tag(rF, vtag);
		map<int, vector<VarOrRVar> >::iterator vit;

		for (vit = vars.begin(); vit != vars.end(); vit++)
		{
			int def = vit->first;
			for (int i = 0; i < vit->second.size(); i++)
			{
				VarOrRVar v = vit->second[i];

				// only add scheduling to defs that are not undef
				if (def == PURE_DEF)
				{
					if (!is_undef(F.values()))
					{
						if (factor)
						{
							F.unroll(v, factor);
							rF.pure_schedule.push_back("unroll(Var(\"" + v.name() + "\")," + std::to_string(factor) + ")");
						}
						else
						{
							F.unroll(v);
							rF.pure_schedule.push_back("unroll(Var(\"" + v.name() + "\"))");
						}
					}
				}
				else
				{
					if (!is_undef(F.update_values(def)))
					{
						if (factor)
						{
							F.update(def).unroll(v, factor);
							rF.update_schedule[def].push_back("unroll(Var(\"" + v.name() + "\")," + std::to_string(factor) + ")");
						}
						else
						{
							F.update(def).unroll(v);
							rF.update_schedule[def].push_back("unroll(Var(\"" + v.name() + "\"))");
						}
					}
				}
			}
		}
	}
	return *this;
}

RecFilterSchedule& RecFilterSchedule::vectorize(VarTag vtag, int factor)
{
	for (int j = 0; j < func_list.size(); j++)
	{
		RecFilterFunc& rF = recfilter.internal_function(func_list[j]);
		Func            F = Func(rF.func);
		// inline済みの皮になる関数はスケジュールなし
		if (rF.func_type == SKIN && rF.external_consumer_func.defined())
		{
			continue;
		}
		map<int, VarOrRVar> vars = var_by_tag(rF, vtag);
		map<int, VarOrRVar>::iterator vit;
		for (vit = vars.begin(); vit != vars.end(); vit++)
		{
			int def = vit->first;
			VarOrRVar v = vit->second;

			// only add scheduling to defs that are not undef

			if (def == PURE_DEF)
			{
				if (!is_undef(F.values()))
				{
					if (factor)
					{
						F.vectorize(v, factor);
						rF.pure_schedule.push_back("vectorize(Var(\"" + v.name() + "\")," + std::to_string(factor) + ")");
					}
					else
					{
						F.vectorize(v);
						rF.pure_schedule.push_back("vectorize(Var(\"" + v.name() + "\"))");
					}
				}
			}
			else
			{
				if (!is_undef(F.update_values(def)))
				{
					if (factor)
					{
						F.update(def).vectorize(v, factor);
						rF.update_schedule[def].push_back("vectorize(Var(\"" + v.name() + "\")," + std::to_string(factor) + ")");
					}
					else
					{
						F.update(def).vectorize(v);
						rF.update_schedule[def].push_back("vectorize(Var(\"" + v.name() + "\"))");
					}
				}
			}
		}
	}
	return *this;
}

RecFilterSchedule& RecFilterSchedule::gpu_blocks(VarTag v1)
{
	return gpu_blocks(v1, VarTag(INVALID), VarTag(INVALID));
}

RecFilterSchedule& RecFilterSchedule::gpu_blocks(VarTag v1, VarTag v2)
{
	return gpu_blocks(v1, v2, VarTag(INVALID));
}

RecFilterSchedule& RecFilterSchedule::gpu_blocks(VarTag v1, VarTag v2, VarTag v3)
{
	if (!recfilter.target().has_gpu_feature())
	{
		cerr << "Cannot use RecFilterSchedule::gpu_blocks() if compilation "
			<< "target is not GPU; use RecFilterSchedule::parallel()" << endl;
		assert(false);
	}

	if (v1.check(SCAN) || v2.check(SCAN) || v3.check(SCAN))
	{
		cerr << "Cannot map a scan variable to parallel GPU blocks" << endl;
		assert(false);
	}

	for (int j = 0; j < func_list.size(); j++)
	{
		RecFilterFunc& rF = recfilter.internal_function(func_list[j]);
		Func            F = Func(rF.func);

		// no need to set gpu_blocks if the function is not compute root
		//if (!rF.func.schedule().compute_level().is_root())
		//{
		//	continue;
		//}

		map<int, VarOrRVar>::iterator vit;
		map<int, vector<VarOrRVar> > parallel_vars;
		map<int, int> next_gpu_block;

		for (int i = 0; i < 3; i++)
		{
			VarTag vtag;

			switch (i)
			{
			case 0: vtag = v1; break;
			case 1: vtag = v2; break;
			case 2: vtag = v3; break;
			default: break;
			}

			if (vtag == INVALID)
			{
				continue;
			}

			map<int, VarOrRVar> vars = var_by_tag(rF, vtag);
			for (vit = vars.begin(); vit != vars.end(); vit++)
			{
				int def = vit->first;
				VarOrRVar v = vit->second.var;
				stringstream s;

				int block_id_x = 0;
				if (next_gpu_block.find(def) != next_gpu_block.end())
				{
					block_id_x = next_gpu_block[def];
					next_gpu_block[def] = block_id_x + 1;
				}
				next_gpu_block[def] = block_id_x + 1;

				if (block_id_x < 0 || block_id_x>2)
				{
					cerr << "Cannot map more than three vars to GPU threads" << endl;
					assert(false);
				}

				s << "parallel(Var(\"" << v.name() << "\")).";
				s << "rename(Var(\"" << v.name() << "\"), Var(\"" << GPU_BLOCK[block_id_x].name() << "\"))";

				// only add scheduling to defs that are not undef
				if (def == PURE_DEF)
				{
					if (!is_undef(F.values()))
					{
						F.parallel(v).rename(v, GPU_BLOCK[block_id_x]);
						rF.pure_schedule.push_back(s.str());
					}
					else
					{
					F.gpu_single_thread();
					rF.pure_schedule.push_back("gpu_single_thread()");
					}
				}
				else
				{
					if (!is_undef(F.update_values(def)))
					{
						F.update(def).parallel(v).rename(v, GPU_BLOCK[block_id_x]);
						rF.update_schedule[def].push_back(s.str());
					}
				}
			}
		}
	}
	return *this;
}

RecFilterSchedule& RecFilterSchedule::gpu_threads(VarTag v1)
{
	return gpu_threads(v1, VarTag(INVALID), VarTag(INVALID));
}

RecFilterSchedule& RecFilterSchedule::gpu_threads(VarTag v1, VarTag v2)
{
	return gpu_threads(v1, v2, VarTag(INVALID));
}

RecFilterSchedule& RecFilterSchedule::gpu_threads(VarTag v1, VarTag v2, VarTag v3)
{

	//if (get_host_target().os == Target::OSX)
	//{
	//	recfilter.setTargetFeature(Target::Metal);
	//}
	//else
	//{
	//	//recfilter.setTargetFeature(Target::CUDA);
	//	recfilter.setTargetFeature(Target::OpenCL);
	//}

	if (!recfilter.target().has_gpu_feature())
	{
		cerr << "Cannot use RecFilterSchedule::gpu_threads() if compilation "
			<< "target is not GPU; use RecFilterSchedule::parallel()" << endl;
		assert(false);
	}

	if (v1.check(SCAN) || v2.check(SCAN) || v3.check(SCAN))
	{
		cerr << "Cannot map a scan variable to parallel threads" << endl;
		assert(false);
	}

	for (int j = 0; j < func_list.size(); j++)
	{
		RecFilterFunc& rF = recfilter.internal_function(func_list[j]);
		Func            F = Func(rF.func);

		map<int, VarOrRVar>::iterator vit;
		map<int, vector<VarOrRVar> > parallel_vars;
		map<int, int> next_gpu_thread;

		for (int i = 0; i < 3; i++)
		{
			VarTag vtag;

			switch (i)
			{
			case 0: vtag = v1; break;
			case 1: vtag = v2; break;
			default:vtag = v3; break;
			}

			if (vtag == INVALID)
			{
				continue;
			}

			map<int, VarOrRVar> vars = var_by_tag(rF, vtag);
			for (vit = vars.begin(); vit != vars.end(); vit++)
			{
				int def = vit->first;
				VarOrRVar v = vit->second.var;
				stringstream s;

				int thread_id_x = 0;
				if (next_gpu_thread.find(def) != next_gpu_thread.end())
				{
					thread_id_x = next_gpu_thread[def];
					next_gpu_thread[def] = thread_id_x + 1;
				}
				next_gpu_thread[def] = thread_id_x + 1;

				if (thread_id_x < 0 || thread_id_x>2)
				{
					cerr << "Cannot map more than three vars to GPU threads" << endl;
					assert(false);
				}

				s << "parallel(Var(\"" << v.name() << "\")).";
				s << "rename(Var(\"" << v.name() << "\"), Var(\"" << GPU_THREAD[thread_id_x].name() << "\"))";

				// only add scheduling to defs that are not undef
				if (def == PURE_DEF)
				{
					if (!is_undef(F.values()))
					{
						F.parallel(v).rename(v, GPU_THREAD[thread_id_x]);
						rF.pure_schedule.push_back(s.str());
					}
				}
				else
				{
					if (!is_undef(F.update_values(def)))
					{
						F.update(def).parallel(v).rename(v, GPU_THREAD[thread_id_x]);
						rF.update_schedule[def].push_back(s.str());
					}
				}
			}
		}
	}
	return *this;
}

RecFilterSchedule& RecFilterSchedule::reorder(VarTag x, VarTag y)
{
	return reorder({ x,y });
}

RecFilterSchedule& RecFilterSchedule::reorder(VarTag x, VarTag y, VarTag z)
{
	return reorder({ x,y,z });
}

RecFilterSchedule& RecFilterSchedule::reorder(VarTag x, VarTag y, VarTag z, VarTag w)
{
	return reorder({ x,y,z,w });
}

RecFilterSchedule& RecFilterSchedule::reorder(VarTag x, VarTag y, VarTag z, VarTag w, VarTag t)
{
	return reorder({ x,y,z,w,t });
}

RecFilterSchedule& RecFilterSchedule::reorder(VarTag x, VarTag y, VarTag z, VarTag w, VarTag s, VarTag t)
{
	return reorder({ x,y,z,w,s,t });
}

RecFilterSchedule& RecFilterSchedule::reorder(VarTag x, VarTag y, VarTag z, VarTag w, VarTag s, VarTag t, VarTag u)
{
	return reorder({ x,y,z,w,s,t,u });
}

RecFilterSchedule& RecFilterSchedule::reorder_storage(VarTag x, VarTag y)
{
	return reorder_storage({ x,y });
}

RecFilterSchedule& RecFilterSchedule::reorder_storage(VarTag x, VarTag y, VarTag z)
{
	return reorder_storage({ x,y,z });
}

RecFilterSchedule& RecFilterSchedule::reorder_storage(VarTag x, VarTag y, VarTag z, VarTag w)
{
	return reorder({ x,y,z,w });
}

RecFilterSchedule& RecFilterSchedule::reorder_storage(VarTag x, VarTag y, VarTag z, VarTag w, VarTag t)
{
	return reorder_storage({ x,y,z,w,t });
}

RecFilterSchedule& RecFilterSchedule::fuse(VarTag vtag1, VarTag vtag2)
{
	for (int j = 0; j < func_list.size(); j++)
	{
		RecFilterFunc& rF = recfilter.internal_function(func_list[j]);
		Func            F = Func(rF.func);

		map<int, VarOrRVar> vars1 = var_by_tag(rF, vtag1);
		map<int, VarOrRVar> vars2 = var_by_tag(rF, vtag2);
		map<int, VarOrRVar>::iterator vit;

		for (vit = vars1.begin(); vit != vars1.end(); vit++)
		{
			int def = vit->first;
			VarOrRVar v1 = vit->second;
			if (vars2.find(def) != vars2.end())
			{
				VarOrRVar v2 = vars2.find(def)->second;

				string s = "fuse(Var(\"" + v1.name() + "\"), Var(\"" + v2.name()
					+ "\"), Var(\"" + v1.name() + "\"))";

				// only add scheduling to defs that are not undef
				if (def == PURE_DEF)
				{
					if (!is_undef(F.values()))
					{
						F.fuse(v1, v2, v1);
						rF.pure_var_category.erase(v2.name());
						rF.pure_schedule.push_back(s);
					}
				}
				else
				{
					if (!is_undef(F.update_values(def)))
					{
						F.update(def).fuse(v1, v2, v1);
						rF.update_var_category[def].erase(v2.name());
						rF.update_schedule[def].push_back(s);
					}
				}
			}
		}
	}
	return *this;
}

RecFilterSchedule& RecFilterSchedule::split(VarTag vtag, int factor)
{
	return split(vtag, factor, INVALID, INVALID);
}

RecFilterSchedule& RecFilterSchedule::split(VarTag vtag, int factor, VarTag var_in)
{
	return split(vtag, factor, var_in, INVALID);
}

RecFilterSchedule& RecFilterSchedule::split(VarTag vtag, int factor, VarTag var_in, VarTag var_out)
{
	if (vtag.check(SPLIT))
	{
		cerr << "Cannot split the variable which was created by split scheduling ops" << endl;
		assert(false);
	}
	if (var_in.check(SCAN) || var_in.check(SPLIT) || var_in.check(FULL) ||
		var_out.check(SCAN) || var_out.check(SPLIT) || var_out.check(FULL))
	{
		cerr << "VarTag for the new dimension created by split must be one of "
			<< "inner(), outer(), inner(i) and outer(k)" << endl;
		assert(false);
	}

	for (int j = 0; j < func_list.size(); j++)
	{
		RecFilterFunc& rF = recfilter.internal_function(func_list[j]); // func_list[j]が名前？
		Func            F = Func(rF.func);

		// inline済みの皮になる関数はスケジュールなし
		if (rF.func_category == REINDEX && rF.external_consumer_func.defined())
		{
			continue;
		}

		// check if the new tag has a count, if not then the use the max possible count
		// afterwards we can use reassign_vartag_counts() to get continuous counts
		// check that vtag_new is not already used
		map<string, VarTag>::iterator z;
		for (z = rF.pure_var_category.begin(); z != rF.pure_var_category.end(); z++)
		{
			if (z->second == var_in || z->second == var_out)
			{
				cerr << "Cannot use " << var_in << " or " << var_out << " as the tag "
					<< "for the new dimesnsion created by split() because "
					<< F.name() << " already has a dimension with the same tag "
					<< "in pure definition" << endl;
				assert(false);
			}
		}
		for (int i = 0; i < rF.update_var_category.size(); i++)
		{
			for (z = rF.update_var_category[i].begin(); z != rF.update_var_category[i].end(); z++)
			{
				if (z->second == var_in || z->second == var_out)
				{
					cerr << "Cannot use " << var_in << " or " << var_out << " as the tag "
						<< "for the new dimesnsion created by split() because "
						<< F.name() << " already has a dimension with the same tag "
						<< "in update definition" << i << endl;
					assert(false);
				}
			}
		}

		// use default tags if they are empty
		VarTag vin = (var_in == INVALID ? vtag | SPLIT : var_in);
		VarTag vout = (var_out == INVALID ? vtag : var_out);

		// add a count if they dont have a count
		vin = (vin == INNER || vin == OUTER ? vin = vin | __4 : vin);
		vout = (vout == INNER || vout == OUTER ? vout = vout | __4 : vout);

		map<int, VarOrRVar> vars = var_by_tag(rF, vtag);
		map<int, VarOrRVar>::iterator vit;

		//int index = vars.size();

		for (vit = vars.begin(); vit != vars.end(); vit++)
		{
			int def = vit->first;
			VarOrRVar v = vit->second.var;
			stringstream s;

			//Var t(unique_name(remove_dollar(v.name())+"1"));
			Var t = Var(v.name() + "i");
			//std::cout << t.name();
			//std::cout << v.name() << ", " << t.name() << endl;

			s << "split(Var(\"" << v.name() << "\"), Var(\"" << v.name()
				<< "\"), Var(\"" << t.name() << "\"), " << factor << ")";
			/*for (Definition d : rF.func.updates())
			{
				for (Expr e : d.args())
				{
					std::cout << e << std::endl;
				}
			}*/

			// only add scheduling to defs that are not undef
			if (def == PURE_DEF)
			{
				F.split(v, v, t, factor, TailStrategy::GuardWithIf);
				rF.pure_var_category.erase(v.name());
				rF.pure_var_category.erase(t.name());
				rF.pure_var_category.insert(make_pair(v.name(), vout));
				rF.pure_var_category.insert(make_pair(t.name(), vin));
				rF.pure_var_splits.insert(make_pair(t.name(), v.name()));
				rF.pure_schedule.push_back(s.str());
				/*std::cout << "pure_var_category...\n";
				for (std::map<string, VarTag>::iterator it = rF.pure_var_category.begin(); it != rF.pure_var_category.end(); it++)
					std::cout << it->first << " " << it->second << std::endl;
				std::cout << std::endl;

				std::cout << "args...\n";
				for (Expr e : rF.func.args())
					std::cout << e << std::endl;
				std::cout << std::endl;

				std::cout << "pure_var_split...\n";
				for (std::map<string, string>::iterator it = rF.pure_var_splits.begin(); it != rF.pure_var_splits.end(); it++)
					std::cout << it->first << " " << it->second << std::endl;
				std::cout << std::endl;*/
				reassign_vartag_counts(rF.pure_var_category, rF.func.args(),
					rF.pure_var_splits);
			}
			else
			{
				F.update(def).split(v, v, t, factor, TailStrategy::GuardWithIf);
				rF.update_var_category[def].erase(v.name());
				rF.update_var_category[def].erase(t.name());
				rF.update_var_category[def].insert(make_pair(v.name(), vout));
				rF.update_var_category[def].insert(make_pair(t.name(), vin));
				rF.update_var_splits[def].insert(make_pair(t.name(), v.name()));
				rF.update_schedule[def].push_back(s.str());

				reassign_vartag_counts(rF.update_var_category[def], rF.func.updates()[def].args(), rF.update_var_splits[def]);
			}
		}
	}
	return *this;
}

RecFilterSchedule& RecFilterSchedule::reorder(vector<VarTag> vtag)
{
	for (int j = 0; j < func_list.size(); j++)
	{
		RecFilterFunc& rF = recfilter.internal_function(func_list[j]);
		Func            F = Func(rF.func);
		// inline済みの皮になる関数はスケジュールなし
		if (rF.func_type == SKIN && rF.external_consumer_func.defined())
		{
			continue;
		}
		map< int, vector<VarOrRVar> > vars;
		map< int, vector<VarOrRVar> >::iterator vit;

		for (int i = 0; i < vtag.size(); i++)
		{
			map< int, vector<VarOrRVar> > vars_x = var_list_by_tag(rF, vtag[i]);
			for (vit = vars_x.begin(); vit != vars_x.end(); vit++)
			{
				for (int k = 0; k < vit->second.size(); k++)
				{
					// check if this var is already in the list
					bool already_added = false;
					for (int u = 0; !already_added && u < vars[vit->first].size(); u++)
					{
						if (vars[vit->first][u].name() == vit->second[k].name())
						{
							already_added = true;
						}
					}
					if (!already_added)
					{
						vars[vit->first].push_back(vit->second[k]);
					}
				}
			}
		}

		for (vit = vars.begin(); vit != vars.end(); vit++)
		{
			int def = vit->first;
			vector<VarOrRVar> var_list = vit->second;
			stringstream s;

			if (var_list.size() > 1) {                 // at least 2 to reorder
				s << "reorder(Internal::vec(";
				for (int i = 0; i < var_list.size(); i++)
				{
					if (i > 0)
					{
						s << ",";
					}
					s << "Var(\"" << var_list[i].name() << "\")";
				}
				s << "))";
				//std::cout << s.str() << std::endl;

				// only add scheduling to defs that are not undef
				if (def == PURE_DEF)
				{
					if (!is_undef(F.values()))
					{
						F.reorder(var_list);
						rF.pure_schedule.push_back(s.str());
					}
				}
				else
				{
					if (!is_undef(F.update_values(def)))
					{
						F.update(def).reorder(var_list);
						rF.update_schedule[def].push_back(s.str());
					}
				}
			}
			else
			{
				cerr << "Warning: VarTags provided to reorder() resulted in less than 2 variables" << endl;
			}
		}
	}
	return *this;
}

RecFilterSchedule& RecFilterSchedule::storage_layout(VarTag innermost, VarTag outermost)
{
	for (int j = 0; j < func_list.size(); j++)
	{
		RecFilterFunc& rF = recfilter.internal_function(func_list[j]);
		Func            F = Func(rF.func);

		// get the list of vars from the pure def
		vector<Var> var_list = F.args();

		// no need to reorder storage
		// - if function is same as final result
		// - if the function is not compute at root level
		// - if less than 2 variables to reorder
		rF.func.lock_loop_levels();
		if (!rF.func.schedule().compute_level().is_root() ||
			rF.func.name() == recfilter.name() || var_list.size() < 2)
		{
			continue;
		}

		// access the variables in the order in which they were defined
		// extract innermost and outermost variables in separate lists
		vector<Var> innermost_var_list;
		vector<Var> middle_var_list;
		vector<Var> outermost_var_list;
		for (int i = 0; i < var_list.size(); i++)
		{
			Var v = var_list[i];
			if (rF.pure_var_category.find(v.name()) != rF.pure_var_category.end())
			{
				VarTag vt = rF.pure_var_category[v.name()];
				if (innermost != INVALID && vt.same_except_count(innermost))
				{
					innermost_var_list.push_back(v);
				}
				else if (outermost != INVALID && vt.same_except_count(outermost))
				{
					outermost_var_list.push_back(v);
				}
				else
				{
					middle_var_list.push_back(v);
				}
			}
		}

		// add the vars in the order of innermost, middle and outermost
		var_list.clear();
		var_list.insert(var_list.end(), innermost_var_list.begin(), innermost_var_list.end());
		var_list.insert(var_list.end(), middle_var_list.begin(), middle_var_list.end());
		var_list.insert(var_list.end(), outermost_var_list.begin(), outermost_var_list.end());

		stringstream s;

		s << "reorder_storage(";
		for (int i = 0; i < var_list.size(); i++)
		{
			if (i > 0)
			{
				s << ",";
			}
			s << "Var(\"" << var_list[i].name() << "\")";
		}
		s << ")";

		switch (var_list.size())
		{
		case 2: F.reorder_storage(
			var_list[0],
			var_list[1]); break;
		case 3: F.reorder_storage(
			var_list[0],
			var_list[1],
			var_list[2]); break;
		case 4: F.reorder_storage(
			var_list[0],
			var_list[1],
			var_list[2],
			var_list[3]); break;
		case 5: F.reorder_storage(
			var_list[0],
			var_list[1],
			var_list[2],
			var_list[3],
			var_list[4]); break;
		default:cerr << "Too many variables in reorder_storage()" << endl; assert(false); break;
		}
		rF.pure_schedule.push_back(s.str());
	}
	return *this;
}

RecFilterSchedule& RecFilterSchedule::reorder_storage(vector<VarTag> vtag)
{
	for (int j = 0; j < func_list.size(); j++)
	{
		RecFilterFunc& rF = recfilter.internal_function(func_list[j]);
		Func            F = Func(rF.func);

		// no need to reorder storage if the function is not compute at root level
		// don't reorder storage of final result
		if (!rF.func.schedule().compute_level().is_root() ||
			rF.func.name() == recfilter.name())
		{
			continue;
		}

		vector<VarOrRVar> var_list;
		map< int, vector<VarOrRVar> >::iterator vit;

		for (int i = 0; i < vtag.size(); i++)
		{
			map< int, vector<VarOrRVar> > vars_x = var_list_by_tag(rF, vtag[i]);
			for (vit = vars_x.begin(); vit != vars_x.end(); vit++)
			{
				if (vit->first == PURE_DEF)
				{
					for (int k = 0; k < vit->second.size(); k++)
					{
						// check if this var is already in the list
						bool already_added = false;
						for (int u = 0; !already_added && u < var_list.size(); u++)
						{
							if (var_list[u].name() == vit->second[k].name())
							{
								already_added = true;
							}
						}
						if (!already_added)
						{
							var_list.push_back(vit->second[k]);
						}
					}
				}
			}
		}

		stringstream s;

		if (var_list.size() > 1) {                 // at least 2 to reorder
			s << "reorder_storage(";
			for (int i = 0; i < var_list.size(); i++)
			{
				if (i > 0)
				{
					s << ",";
				}
				s << "Var(\"" << var_list[i].name() << "\")";
			}
			s << ")";

			switch (var_list.size())
			{
			case 2: F.reorder_storage(
				var_list[0].var,
				var_list[1].var); break;
			case 3: F.reorder_storage(
				var_list[0].var,
				var_list[1].var,
				var_list[2].var); break;
			case 4: F.reorder_storage(
				var_list[0].var,
				var_list[1].var,
				var_list[2].var,
				var_list[3].var); break;
			case 5: F.reorder_storage(
				var_list[0].var,
				var_list[1].var,
				var_list[2].var,
				var_list[3].var,
				var_list[4].var); break;
			default:cerr << "Too many variables in reorder_storage()" << endl; assert(false); break;
			}
			rF.pure_schedule.push_back(s.str());
		}
		else
		{
			cerr << "Warning: VarTags provided to reorder_storage() resulted in less than 2 variables" << endl;
		}
	}
	return *this;
}
