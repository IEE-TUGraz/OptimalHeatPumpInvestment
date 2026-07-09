"""
Out-of-sample (OOS) ex-post validation.

The in-sample ex-post analysis (`expost_model.solve_expost_model`, driven by
`expost_analysis.run_kpi_expost_analysis`) re-optimises operation on the
*aggregated* representative periods and measures how well the part-load
approximation of each formulation holds. This module does the complementary
*out-of-sample* validation: for every scenario it takes each formulation's
**fixed investment** and re-solves the same operational 9-segment ex-post
model, but on the **full-resolution chronological time series** (the original
data over the scenario's StartDay/DurationDays window).

Because the full series (tens of thousands of quarter-hours) is too large for a
single MILP, it is solved as a rolling horizon: consecutive chunks solved in
order, with the storage level and the heat-pump on/off state handed over from
one chunk to the next (linked storage). Each chunk is solved with a small
look-ahead overlap of which only the leading part is committed, to avoid the
end-of-chunk storage-drain artefact of a naive split.

Standalone / on demand: see `run_oos_expost_analysis` and the `__main__` block.
Investment stays fixed, so every chunk is a small, fast operational MILP.
"""

import os
import json
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pyomo.environ as pyo

import data
import Core_model
import expost_analysis


# 9-segment part-load COP breakpoints (identical to expost_model.solve_expost_model)
_SEGMENTS = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9']
_K = {'s1': 3.633, 's2': 3.704, 's3': 3.740, 's4': 3.68, 's5': 3.666,
      's6': 3.68, 's7': 3.678, 's8': 3.664, 's9': 3.632}
_D_POS = {s: 0.0 for s in _SEGMENTS}
_D_NEG = {'s1': 0.1118, 's2': 0.126, 's3': 0.1392, 's4': 0.112, 's5': 0.105,
          's6': 0.1134, 's7': 0.112, 's8': 0.1008, 's9': 0.072}
_R_MIN = {'s1': 0.1, 's2': 0.2, 's3': 0.3, 's4': 0.4, 's5': 0.5,
          's6': 0.6, 's7': 0.7, 's8': 0.8, 's9': 0.9}
_R_MAX = {'s1': 0.2, 's2': 0.3, 's3': 0.4, 's4': 0.5, 's5': 0.6,
          's6': 0.7, 's7': 0.8, 's8': 0.9, 's9': 1.0}


########################################################################################################################
## Chunk preparation
########################################################################################################################

def _as_rp_frame(df):
    """Turn a time_h-indexed frame into a (rp, time_h) MultiIndex frame (single rp)."""
    out = df.copy()
    out.index = pd.MultiIndex.from_arrays(
        [['rp01'] * len(out), df.index],
        names=['rp', 'time_h'],
    )
    return out


########################################################################################################################
## Operational chunk model (fixed investment, linked storage / startup state)
########################################################################################################################

def _solve_operational_chunk(
    parameter,
    global_param,
    df_heat_demand,
    df_el_price,
    df_cop_scalor,
    investment,
    initial_storage,
    initial_on_state,
    keep_labels,
    mipgap=1e-4,  # Gurobi default; ex-post is the validation reference, keep it tight
    threads=None,
):
    """
    Solve the operational 9-segment ex-post model for one chronological chunk.

    Investment is fixed to `investment` ({"HP1": kW, "HP2": kW}). Storage starts
    at `initial_storage` (kWh) and the heat pumps start in state `initial_on_state`
    ({"HP1": 0/1, "HP2": 0/1}); neither is cyclic. Costs and the returned time
    series cover only `keep_labels` (the committed, non-overlap part), while the
    solved window may extend beyond them for look-ahead.

    Returns (shares, df_results_kept, final_storage, final_on_state).
    """
    df_rp_weights = pd.DataFrame({'weight': [1]}, index=['rp01'])

    model = pyo.ConcreteModel()

    # --- Sets ---
    Core_model.initialise_sets(model, df_heat_demand)
    model.s = pyo.Set(initialize=_SEGMENTS)

    # --- Parameters ---
    Core_model.initialise_vector_parameters(model, df_el_price, df_heat_demand, df_cop_scalor, df_rp_weights)
    Core_model.initialise_scalar_parameters(model, parameter, global_param)

    # investment fixed -> upper/lower bound params both equal the fixed capacity
    model.InvestmentUB = pyo.Param(model.hps, initialize={"HP1": investment["HP1"], "HP2": investment["HP2"]})
    model.InvestmentLB = pyo.Param(model.hps, initialize={"HP1": investment["HP1"], "HP2": investment["HP2"]})
    model.MinPLR = pyo.Param(initialize=parameter["MinPartLoad"])

    # 9-segment part-load parameters
    model.K = pyo.Param(model.s, initialize=_K)
    model.D_pos = pyo.Param(model.s, initialize=_D_POS)
    model.D_neg = pyo.Param(model.s, initialize=_D_NEG)
    model.RMin = pyo.Param(model.s, initialize=_R_MIN)
    model.RMax = pyo.Param(model.s, initialize=_R_MAX)

    # --- Variables ---
    Core_model.initialise_variables(model)
    for hps in model.hps:
        model.hp_invest[hps].fix(investment[hps])
        model.hp_base_invest[hps].fix(1 if investment[hps] > 0 else 0)

    model.z_PL = pyo.Var(model.rp, model.h, model.s, model.hps, within=pyo.Binary)
    model.d_scaled_pos = pyo.Var(model.rp, model.h, model.s, model.hps, within=pyo.NonNegativeReals)
    model.d_scaled_neg = pyo.Var(model.rp, model.h, model.s, model.hps, within=pyo.NonNegativeReals)
    model.p_elec_seg = pyo.Var(model.rp, model.h, model.s, model.hps, within=pyo.NonNegativeReals)

    Core_model.initialise_binary_variables(model)

    # --- Storage (linked, non-cyclic) ---
    _add_linked_storage(model, initial_storage)

    # --- Heat pump 9-segment COP constraints (mirror expost_model) ---
    def uc_cop_constraint_rule(m, rp, h, hps):
        return m.q_heat[rp, h, hps] == sum(
            (m.K[s] * m.p_elec_seg[rp, h, s, hps] + m.d_scaled_pos[rp, h, s, hps] - m.d_scaled_neg[rp, h, s, hps])
            for s in m.s
        ) * m.COP_Scalor[rp, h]
    model.COPUCConstraint = pyo.Constraint(model.rp, model.h, model.hps, rule=uc_cop_constraint_rule)

    def intercept_active_segment_rule_pos(m, rp, h, s, hps):
        return m.d_scaled_pos[rp, h, s, hps] <= m.z_PL[rp, h, s, hps] * m.D_pos[s] * m.InvestmentUB[hps]
    model.InterceptActiveSegmentConstraint_pos = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=intercept_active_segment_rule_pos)

    def intercept_upper_bound_rule_pos(m, rp, h, s, hps):
        return m.d_scaled_pos[rp, h, s, hps] <= m.D_pos[s] * m.hp_invest[hps]
    model.InterceptUpperBoundConstraint_pos = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=intercept_upper_bound_rule_pos)

    def intercept_lower_bound_rule_pos(m, rp, h, s, hps):
        return m.d_scaled_pos[rp, h, s, hps] >= m.D_pos[s] * m.hp_invest[hps] - (1 - m.z_PL[rp, h, s, hps]) * m.InvestmentUB[hps] * m.D_pos[s]
    model.InterceptLowerBoundConstraint_pos = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=intercept_lower_bound_rule_pos)

    def intercept_active_segment_rule_neg(m, rp, h, s, hps):
        return m.d_scaled_neg[rp, h, s, hps] <= m.z_PL[rp, h, s, hps] * m.D_neg[s] * m.InvestmentUB[hps]
    model.InterceptActiveSegmentConstraint_neg = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=intercept_active_segment_rule_neg)

    def intercept_upper_bound_rule_neg(m, rp, h, s, hps):
        return m.d_scaled_neg[rp, h, s, hps] <= m.D_neg[s] * m.hp_invest[hps]
    model.InterceptUpperBoundConstraint_neg = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=intercept_upper_bound_rule_neg)

    def intercept_lower_bound_rule_neg(m, rp, h, s, hps):
        return m.d_scaled_neg[rp, h, s, hps] >= m.D_neg[s] * m.hp_invest[hps] - (1 - m.z_PL[rp, h, s, hps]) * m.InvestmentUB[hps] * m.D_neg[s]
    model.InterceptLowerBoundConstraint_neg = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=intercept_lower_bound_rule_neg)

    def active_segment_rule(m, rp, h, hps):
        return sum(m.z_PL[rp, h, s, hps] for s in m.s) == m.z_hp[rp, h, hps]
    model.ActiveSegmentConstraint = pyo.Constraint(model.rp, model.h, model.hps, rule=active_segment_rule)

    def segment_max_production_rule(m, rp, h, s, hps):
        return m.p_elec_seg[rp, h, s, hps] <= m.z_PL[rp, h, s, hps] * m.InvestmentUB[hps]
    model.SegmentProductionConstraint = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=segment_max_production_rule)

    def segment_max_invest_rule(m, rp, h, s, hps):
        return m.p_elec_seg[rp, h, s, hps] <= m.hp_invest[hps] * m.RMax[s]
    model.SegmentInvestmentConstraint = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=segment_max_invest_rule)

    def segment_min_production_rule(m, rp, h, s, hps):
        return m.p_elec_seg[rp, h, s, hps] >= m.hp_invest[hps] * m.RMin[s] - (1 - m.z_PL[rp, h, s, hps]) * m.InvestmentUB[hps]
    model.SegmentMinProductionConstraint = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=segment_min_production_rule)

    def actual_production_rule(m, rp, h, hps):
        return m.p_el[rp, h, hps] == sum(m.p_elec_seg[rp, h, s, hps] for s in m.s)
    model.ActualProductionConstraint = pyo.Constraint(model.rp, model.h, model.hps, rule=actual_production_rule)

    def uc_max_production_rule(m, rp, h, hps):
        return m.p_el[rp, h, hps] <= m.z_hp[rp, h, hps] * m.InvestmentUB[hps]
    model.UCProductionLimit = pyo.Constraint(model.rp, model.h, model.hps, rule=uc_max_production_rule)

    def uc_min_production_rule(m, rp, h, hps):
        return m.p_el[rp, h, hps] >= m.hp_invest[hps] * m.MinPLR - (1 - m.z_hp[rp, h, hps]) * m.InvestmentUB[hps] * m.MinPLR
    model.UCMinProductionLimit = pyo.Constraint(model.rp, model.h, model.hps, rule=uc_min_production_rule)

    # cap electricity by fixed capacity (investment is fixed, so this is just a bound)
    def max_invest_rule(m, rp, h, hps):
        return m.p_el[rp, h, hps] <= m.hp_invest[hps]
    model.MaxInvestConstraint = pyo.Constraint(model.rp, model.h, model.hps, rule=max_invest_rule)

    startup_on = str(parameter["StartUpConstraint"]).strip().lower() == "true"
    if startup_on:
        _add_linked_startup(model, initial_on_state)

    # --- Objective: operational costs only (investment added once at aggregation) ---
    _add_operational_objective(model, startup_on)

    # --- Solve ---
    solver = pyo.SolverFactory('gurobi_persistent')
    solver.set_instance(model)
    solver.options["MIPGap"] = mipgap
    if threads is not None:
        solver.options["Threads"] = threads  # cap cores per solve to avoid oversubscription when run in parallel
    solver.solve(tee=False)

    # --- Extract results for the committed (kept) part only ---
    shares = _collect_shares(model, keep_labels, startup_on)
    df_kept = _collect_timeseries(model, keep_labels)
    final_storage, final_on_state = _handoff_state(model, keep_labels[-1])

    return shares, df_kept, final_storage, final_on_state


def _add_linked_storage(model, initial_storage):
    model.InitStorage = pyo.Param(initialize=float(initial_storage))

    def power_balance_rule(m, rp, h):
        return m.HeatDemand[rp, h] == (
            sum(m.q_heat[rp, h, hps] for hps in m.hps)
            + m.heat_storage_discharge[rp, h]
            - m.heat_storage_charge[rp, h]
            + m.heat_not_supplied[rp, h]
            - m.excess_heat_supplied[rp, h]
        )
    model.PowerBalance = pyo.Constraint(model.rp, model.h, rule=power_balance_rule)

    def storage_balance_rule(m, rp, h):
        if h == m.h.first():
            return m.heat_storage_level[rp, h] == m.InitStorage
        h_prev = m.h.prev(h)
        return m.heat_storage_level[rp, h] == (
            m.heat_storage_level[rp, h_prev] * (1 - m.StorageLoss * m.DeltaH)
            + (m.heat_storage_charge[rp, h_prev] * m.StorageChEff - m.heat_storage_discharge[rp, h_prev]) * m.DeltaH
        )
    model.HeatStorageBalance = pyo.Constraint(model.rp, model.h, rule=storage_balance_rule)


def _add_linked_startup(model, initial_on_state):
    model.InitOn = pyo.Param(model.hps, initialize={hps: int(initial_on_state[hps]) for hps in ["HP1", "HP2"]})

    def startup_rule(m, rp, h, hps):
        prev = m.InitOn[hps] if h == m.h.first() else m.z_hp[rp, m.h.prev(h), hps]
        return m.startup_hp[rp, h, hps] >= m.z_hp[rp, h, hps] - prev
    model.StartupDefinition = pyo.Constraint(model.rp, model.h, model.hps, rule=startup_rule)


def _add_operational_objective(model, startup_on):
    def objective_rule(m):
        cost = (
            sum(m.ElectricityCost[rp, h] * m.p_el[rp, h, hps] * m.DeltaH * m.rpWeight[rp]
                for rp in m.rp for h in m.h for hps in m.hps)
            + sum(m.Cost_HeatNotSupplied * m.heat_not_supplied[rp, h] * m.DeltaH * m.rpWeight[rp]
                  for rp in m.rp for h in m.h)
            + sum(m.Cost_ExcessHeat * m.excess_heat_supplied[rp, h] * m.DeltaH * m.rpWeight[rp]
                  for rp in m.rp for h in m.h)
        )
        if startup_on:
            cost += sum(m.StartupCosts * m.startup_hp[rp, h, hps] * m.rpWeight[rp]
                        for rp in m.rp for h in m.h for hps in m.hps)
        return cost
    model.TotalCost = pyo.Objective(rule=objective_rule, sense=pyo.minimize)


def _collect_shares(model, keep_labels, startup_on):
    rp = 'rp01'
    dh = pyo.value(model.DeltaH)
    el = sum(pyo.value(model.ElectricityCost[rp, h]) * pyo.value(model.p_el[rp, h, hps]) * dh
             for h in keep_labels for hps in model.hps)
    hns = sum(pyo.value(model.Cost_HeatNotSupplied) * pyo.value(model.heat_not_supplied[rp, h]) * dh
              for h in keep_labels)
    ehs = sum(pyo.value(model.Cost_ExcessHeat) * pyo.value(model.excess_heat_supplied[rp, h]) * dh
              for h in keep_labels)
    startup = 0.0
    if startup_on:
        startup = sum(pyo.value(model.StartupCosts) * pyo.value(model.startup_hp[rp, h, hps])
                      for h in keep_labels for hps in model.hps)
    hns_energy = sum(pyo.value(model.heat_not_supplied[rp, h]) * dh for h in keep_labels)
    ehs_energy = sum(pyo.value(model.excess_heat_supplied[rp, h]) * dh for h in keep_labels)
    return {
        "ElectricityCosts": el,
        "HeatNotSuppliedCosts": hns,
        "ExcessHeatCosts": ehs,
        "StartupCosts": startup,
        "Heat_Not_Supplied_Total": hns_energy,
        "Excess_Heat_Supplied_Total": ehs_energy,
    }


def _collect_timeseries(model, keep_labels):
    rp = 'rp01'
    rows = {}
    for h in keep_labels:
        row = {"HeatDemand": pyo.value(model.HeatDemand[rp, h])}
        for hps in model.hps:
            row[f"Electricity_Consumption_kW_{hps}"] = pyo.value(model.p_el[rp, h, hps])
            row[f"Heat_Produced_kW_{hps}"] = pyo.value(model.q_heat[rp, h, hps])
        row["Heat_Storage_Level_kWh"] = pyo.value(model.heat_storage_level[rp, h])
        row["Heat_Storage_Charge_kW"] = pyo.value(model.heat_storage_charge[rp, h])
        row["Heat_Storage_Discharge_kW"] = pyo.value(model.heat_storage_discharge[rp, h])
        row["Heat_Not_Supplied_kW"] = pyo.value(model.heat_not_supplied[rp, h])
        row["Excess_Heat_Supplied_kW"] = pyo.value(model.excess_heat_supplied[rp, h])
        rows[h] = row
    df = pd.DataFrame.from_dict(rows, orient='index')
    df.index.name = 'time_h'
    return df


def _handoff_state(model, last_kept_label):
    """State entering the next chunk: storage level and on/off after the last kept step."""
    rp = 'rp01'
    h = last_kept_label
    level = pyo.value(model.heat_storage_level[rp, h])
    loss = pyo.value(model.StorageLoss)
    dh = pyo.value(model.DeltaH)
    ch_eff = pyo.value(model.StorageChEff)
    cap = pyo.value(model.StorageCap)
    charge = pyo.value(model.heat_storage_charge[rp, h])
    discharge = pyo.value(model.heat_storage_discharge[rp, h])
    next_level = level * (1 - loss * dh) + (charge * ch_eff - discharge) * dh
    next_level = min(max(next_level, 0.0), cap)
    on_state = {hps: int(round(pyo.value(model.z_hp[rp, h, hps]))) for hps in model.hps}
    return next_level, on_state


########################################################################################################################
## Rolling-horizon driver over the full chronological series
########################################################################################################################

def solve_oos_full_series(
    parameter,
    df_heat_demand_full,
    df_el_price_full,
    df_cop_scalor_full,
    investment,
    chunk_steps=672,
    overlap_steps=96,
    initial_storage=0.0,
    mipgap=1e-4,  # Gurobi default; ex-post is the validation reference, keep it tight
    threads=None,
    label="",
):
    """
    Solve the operational model over the full chronological series for a fixed
    investment, as a rolling horizon with linked storage / startup state.

    chunk_steps: committed steps per chunk (672 = 1 week at 15-min resolution).
    overlap_steps: additional look-ahead steps solved but not committed.

    Returns (results_dict, df_timeseries).
    """
    global_param = data.load_parameter()
    n = len(df_heat_demand_full)

    agg = {
        "ElectricityCosts": 0.0,
        "HeatNotSuppliedCosts": 0.0,
        "ExcessHeatCosts": 0.0,
        "StartupCosts": 0.0,
        "Heat_Not_Supplied_Total": 0.0,
        "Excess_Heat_Supplied_Total": 0.0,
    }
    ts_parts = []

    storage = float(initial_storage)
    on_state = {"HP1": 0, "HP2": 0}

    start = 0
    n_chunks = 0
    while start < n:
        keep_end = min(start + chunk_steps, n)
        solve_end = min(keep_end + overlap_steps, n)

        hd = _as_rp_frame(df_heat_demand_full.iloc[start:solve_end])
        ep = _as_rp_frame(df_el_price_full.iloc[start:solve_end])
        cop = _as_rp_frame(df_cop_scalor_full.iloc[start:solve_end])

        keep_labels = df_heat_demand_full.index[start:keep_end].tolist()

        shares, df_kept, storage, on_state = _solve_operational_chunk(
            parameter, global_param, hd, ep, cop,
            investment, storage, on_state, keep_labels, mipgap=mipgap, threads=threads,
        )

        for k in agg:
            agg[k] += shares[k]
        ts_parts.append(df_kept)

        n_chunks += 1
        print(f"    {label}chunk {n_chunks}: steps [{start}:{keep_end}] "
              f"(solved to {solve_end}), storage handoff {storage:.2f} kWh")

        start = keep_end

    # --- Investment cost: counted once over the whole horizon ---
    time_weight = parameter["DurationDays"] / 365.0
    invest_costs = sum(investment[hps] * parameter["InvestmentCost"] * time_weight for hps in ["HP1", "HP2"])
    invest_costs += parameter["BaseInvestmentCost"] * sum(1 for hps in ["HP1", "HP2"] if investment[hps] > 0) * time_weight

    results_dict = {
        "HP_Investment_HP1": investment["HP1"],
        "HP_Investment_HP2": investment["HP2"],
        "InvestmentCosts": invest_costs,
        "ElectricityCosts": agg["ElectricityCosts"],
        "HeatNotSuppliedCosts": agg["HeatNotSuppliedCosts"],
        "ExcessHeatCosts": agg["ExcessHeatCosts"],
        "StartupCosts": agg["StartupCosts"],
        "TotalCost_ex_post": (
            invest_costs + agg["ElectricityCosts"] + agg["HeatNotSuppliedCosts"]
            + agg["ExcessHeatCosts"] + agg["StartupCosts"]
        ),
        "Heat_Not_Supplied_Total": agg["Heat_Not_Supplied_Total"],
        "Excess_Heat_Supplied_Total": agg["Excess_Heat_Supplied_Total"],
        "n_chunks": n_chunks,
        "chunk_steps": chunk_steps,
        "overlap_steps": overlap_steps,
    }

    df_timeseries = pd.concat(ts_parts)
    return results_dict, df_timeseries


########################################################################################################################
## Scenario-level entry point
########################################################################################################################

def _run_one_model(task):
    """
    Solve the full-series OOS for a single formulation and persist its results.
    Top-level (picklable) so it can run in a separate process. The chunks within
    a formulation stay sequential, so linked storage is preserved; only distinct
    formulations run concurrently. Returns a KPI row dict (or None if skipped).
    """
    (model, scenario_index, scenario_params, df_hd_full, df_ep_full, df_cop_full,
     out_dir, chunk_steps, overlap_steps, initial_storage, mipgap, threads, source_index) = task

    key_results = expost_analysis.load_key_results(source_index, model)
    if key_results is None:
        print(f"  [OOS] skipping {model}: no key results.")
        return None

    investment = {"HP1": key_results["HP_Investment_HP1"], "HP2": key_results["HP_Investment_HP2"]}
    print(f"  [OOS] {model}: fixed investment HP1={investment['HP1']:.2f} kW, "
          f"HP2={investment['HP2']:.2f} kW over {len(df_hd_full)} steps.")

    results_dict, df_ts = solve_oos_full_series(
        scenario_params, df_hd_full, df_ep_full, df_cop_full,
        investment, chunk_steps=chunk_steps, overlap_steps=overlap_steps,
        initial_storage=initial_storage, mipgap=mipgap, threads=threads,
        label=f"[{model}] ",
    )

    df_ts.to_excel(os.path.join(out_dir, f"expost_oos_{model}_results.xlsx"))
    with open(os.path.join(out_dir, f"expost_oos_{model}_key_results.json"), "w") as f:
        json.dump(results_dict, f, indent=4)

    print(f"  [OOS] {model}: TotalCost_ex_post = {results_dict['TotalCost_ex_post']:.2f}")
    return {"scenario": scenario_index, "model": model, **results_dict}


def run_oos_expost_analysis(
    scenario_index,
    scenario_params,
    model_data,
    models=("LP", "UC", "PWL", "PWLR", "CR"),
    chunk_steps=672,
    overlap_steps=96,
    initial_storage=0.0,
    mipgap=1e-4,  # Gurobi default; ex-post is the validation reference, keep it tight
    n_jobs=None,
    threads_per_job=None,
    source_index=None,
    results_root="results",
):
    """
    Run the out-of-sample ex-post validation for one scenario, over all
    formulations. Fixed investments are read from each formulation's in-sample
    `<MODEL>_key_results.json` in `results/scenario_<source_index>/`
    (defaults to `scenario_index`). Writes per-formulation
    `expost_oos_<MODEL>_results.xlsx` / `_key_results.json` and returns a KPI
    DataFrame.

    Formulations are independent and run concurrently across `n_jobs` processes
    (default: one per formulation, capped at the CPU count); the chunks within a
    formulation stay sequential so linked storage is preserved. `threads_per_job`
    caps Gurobi threads per process to avoid core oversubscription (default:
    total cores split evenly over the jobs).
    """
    source_index = scenario_index if source_index is None else source_index

    # Full-resolution chronological data for the scenario window (from the pickle).
    df_hd_full, df_ep_full, df_cop_full = data.extract_relevant_data(model_data, scenario_params)

    out_dir = os.path.join(results_root, f"scenario_{scenario_index}")
    os.makedirs(out_dir, exist_ok=True)

    n_cpu = os.cpu_count() or 1
    if n_jobs is None:
        n_jobs = min(len(models), n_cpu)
    n_jobs = max(1, min(n_jobs, len(models)))
    if threads_per_job is None:
        threads_per_job = max(1, n_cpu // n_jobs) if n_jobs > 1 else None

    tasks = [
        (model, scenario_index, scenario_params, df_hd_full, df_ep_full, df_cop_full,
         out_dir, chunk_steps, overlap_steps, initial_storage, mipgap, threads_per_job, source_index)
        for model in models
    ]

    print(f"  [OOS] scenario {scenario_index}: {len(tasks)} formulations, "
          f"n_jobs={n_jobs}, threads/job={threads_per_job}, chunk_steps={chunk_steps}, mipgap={mipgap}")

    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            rows = list(executor.map(_run_one_model, tasks))
    else:
        rows = [_run_one_model(t) for t in tasks]

    rows = [r for r in rows if r is not None]
    return pd.DataFrame(rows)


def run_oos_for_all_scenarios(
    scenarios_path="inputScenarios.xlsx",
    data_path="IBK_case_study_data.pkl",
    output_path="results/All_KPI_Results_OOS.xlsx",
    **kwargs,
):
    """Run the OOS analysis for every scenario in the scenarios file and aggregate KPIs."""
    scenarios = data.load_input_scenarios(scenarios_path)
    model_data = data.load_data(data_path)

    all_rows = []
    for _, scenario in scenarios.iterrows():
        params = scenario.to_dict()
        print(f"\n=== OOS ex-post for scenario {params['ScenarioIndex']} ===")
        df_kpi = run_oos_expost_analysis(
            params["ScenarioIndex"], params, model_data, **kwargs,
        )
        all_rows.append(df_kpi)

    final_df = pd.concat(all_rows, ignore_index=True)
    final_df.to_excel(output_path, index=False)
    print(f"\nAggregated OOS KPIs written to {output_path}")
    return final_df


if __name__ == "__main__":
    run_oos_for_all_scenarios()
