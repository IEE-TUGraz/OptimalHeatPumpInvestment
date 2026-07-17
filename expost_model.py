import pandas as pd
import pyomo.environ as pyo
import os
import data
import Core_model

def solve_expost_model(parameter, df_heat_demand, df_el_price, df_cop_scalor, df_rp_weights, dict_key_results, model_name):
    global_param = data.load_parameter()
    model = pyo.ConcreteModel()

    rel = parameter["InvBoundRange"]


    # --- Sets ---
    Core_model.initialise_sets(model, df_heat_demand)
    model.s = pyo.Set(initialize=['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9'])  # segments for piecewise linear COP

    # --- Parameters ---
    # vectors
    Core_model.initialise_vector_parameters(model, df_el_price, df_heat_demand, df_cop_scalor, df_rp_weights)

    # scalars
    Core_model.initialise_scalar_parameters(model, parameter, global_param)
    #Core_model.initialise_investment_bounds(model, parameter, dict_key_results)
    model.InvestmentUB = pyo.Param(model.hps, initialize={"HP1": dict_key_results["HP_Investment_HP1"], "HP2": dict_key_results["HP_Investment_HP2"]},
                                   mutable=False)  # invest_estimate * (1 + rel), mutable=True)  # kW
    model.InvestmentLB = pyo.Param(model.hps, initialize={"HP1": dict_key_results["HP_Investment_HP1"], "HP2": dict_key_results["HP_Investment_HP2"]},
                                   mutable=False)

    model.MinPLR = pyo.Param(initialize=parameter["MinPartLoad"])  # minimum part load ratio

    # for piecewise linear COP, define breakpoints and slopes
    model.z_PL = pyo.Var(model.rp, model.h, model.s, model.hps, within=pyo.Binary)  # segment selection variable
    model.d_scaled_pos = pyo.Var(model.rp, model.h, model.s, model.hps, within=pyo.NonNegativeReals)
    model.d_scaled_neg = pyo.Var(model.rp, model.h, model.s, model.hps, within=pyo.NonNegativeReals)

    model.K = pyo.Param(model.s, initialize={'s1': 3.633, 's2': 3.704, 's3': 3.748, 's4': 3.68, 's5': 3.666, 's6': 3.68, 's7': 3.678, 's8': 3.664, 's9': 3.632})  # segment slope (s3 corrected 3.740->3.748 to match datasheet secant)
    model.D_pos = pyo.Param(model.s, initialize={'s1': 0, 's2': 0, 's3': 0.0, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0, 's9': 0})
    model.D_neg = pyo.Param(model.s, initialize={'s1': 0.1118, 's2': 0.126, 's3': 0.1392, 's4': 0.112, 's5': 0.105, 's6': 0.1134, 's7': 0.112, 's8': 0.1008, 's9': 0.072})
    model.RMin = pyo.Param(model.s, initialize={'s1': 0.1, 's2': 0.2, 's3': 0.3, 's4': 0.4, 's5': 0.5, 's6': 0.6, 's7': 0.7, 's8': 0.8, 's9': 0.9})  # segment minimum PLR
    model.RMax = pyo.Param(model.s, initialize={'s1': 0.2, 's2': 0.3, 's3': 0.4, 's4': 0.5, 's5': 0.6, 's6': 0.7, 's7': 0.8, 's8': 0.9, 's9': 1.0})  # segment maximum PLR

    # --- Variables ---
    Core_model.initialise_variables(model)

    # fix the investment to the values obtained from the previous optimisation
    for hps in model.hps:
        model.hp_invest[hps].fix(dict_key_results[f"HP_Investment_{hps}"])

    # for piecewise linear COP
    model.p_elec_seg = pyo.Var(model.rp, model.h, model.s, model.hps, within=pyo.NonNegativeReals)  # electricity consumption per segment

    # binaries
    Core_model.initialise_binary_variables(model)

    # --- Model ---
    # add the storage formulation
    Core_model.add_storage_formulation(model)

    # Heat Pump Constraints
    def uc_cop_constraint_rule(m, rp, h, hps):
        return m.q_heat[rp, h, hps] == sum((m.K[s] * m.p_elec_seg[rp, h, s, hps] + m.d_scaled_pos[rp, h, s, hps] - m.d_scaled_neg[rp, h,s, hps]) for s in m.s) * m.COP_Scalor[rp, h]
    model.COPUCConstraint = pyo.Constraint(model.rp, model.h, model.hps, rule=uc_cop_constraint_rule)

    Core_model.add_investment_formulation(model)

    def intercept_active_segment_rule_pos(m, rp, h, s, hps):
        return m.d_scaled_pos[rp, h,s, hps] <= m.z_PL[rp, h,s, hps] * m.D_pos[s] * m.InvestmentUB[hps]
    model.InterceptActiveSegmentConstraint_pos = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=intercept_active_segment_rule_pos)

    def intercept_upper_bound_rule_pos(m, rp, h, s, hps):
        return m.d_scaled_pos[rp, h,s, hps] <= m.D_pos[s] * m.hp_invest[hps]
    model.InterceptUpperBoundConstraint_pos = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=intercept_upper_bound_rule_pos)

    def intercept_lower_bound_rule_pos(m, rp, h, s, hps):
        return m.d_scaled_pos[rp, h,s, hps] >= m.D_pos[s] * m.hp_invest[hps] - (1 - m.z_PL[rp, h,s, hps]) * m.InvestmentUB[hps] * m.D_pos[s]
    model.InterceptLowerBoundConstraint_pos = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=intercept_lower_bound_rule_pos)

    def intercept_active_segment_rule_neg(m, rp, h, s, hps):
        return m.d_scaled_neg[rp, h,s, hps] <= m.z_PL[rp, h,s, hps] * m.D_neg[s] * m.InvestmentUB[hps]
    model.InterceptActiveSegmentConstraint_neg = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=intercept_active_segment_rule_neg)

    def intercept_upper_bound_rule_neg(m, rp, h, s, hps):
        return m.d_scaled_neg[rp, h,s, hps] <= m.D_neg[s] * m.hp_invest[hps]
    model.InterceptUpperBoundConstraint_neg = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=intercept_upper_bound_rule_neg)

    def intercept_lower_bound_rule_neg(m, rp, h, s, hps):
        return m.d_scaled_neg[rp, h,s, hps] >= m.D_neg[s] * m.hp_invest[hps] - (1 - m.z_PL[rp, h,s, hps]) * m.InvestmentUB[hps] * m.D_neg[s]
    model.InterceptLowerBoundConstraint_neg = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=intercept_lower_bound_rule_neg)

    def active_segment_rule(m, rp, h, hps):
        return sum(m.z_PL[rp, h,s, hps] for s in m.s) == m.z_hp[rp, h, hps]
    model.ActiveSegmentConstraint = pyo.Constraint(model.rp, model.h, model.hps, rule=active_segment_rule)

    def segment_max_production_rule(m, rp, h, s, hps):
        return m.p_elec_seg[rp, h,s, hps] <= m.z_PL[rp, h,s, hps] * m.InvestmentUB[hps]
    model.SegmentProductionConstraint = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=segment_max_production_rule)

    def segment_max_invest_rule(m, rp, h, s, hps):
        return m.p_elec_seg[rp, h,s, hps] <= m.hp_invest[hps] * m.RMax[s]
    model.SegmentInvestmentConstraint = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=segment_max_invest_rule)

    def segment_min_production_rule(m, rp, h, s, hps):
        return m.p_elec_seg[rp, h,s, hps] >= m.hp_invest[hps] * m.RMin[s] - (1 - m.z_PL[rp, h,s, hps]) * m.InvestmentUB[hps]
    model.SegmentMinProductionConstraint = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=segment_min_production_rule)

    def actual_production_rule(m, rp, h, hps):
        return m.p_el[rp, h, hps] == sum(m.p_elec_seg[rp, h,s, hps] for s in m.s)
    model.ActualProductionConstraint = pyo.Constraint(model.rp, model.h, model.hps, rule=actual_production_rule)

    def uc_max_production_rule(m, rp, h, hps):
        return m.p_el[rp, h, hps] <= m.z_hp[rp, h, hps] * m.InvestmentUB[hps]
    model.UCProductionLimit = pyo.Constraint(model.rp, model.h, model.hps, rule=uc_max_production_rule)

    def uc_min_production_rule(m, rp, h, hps):
        return m.p_el[rp, h, hps] >= m.hp_invest[hps] * m.MinPLR - (1 - m.z_hp[rp, h, hps]) * m.InvestmentUB[hps] * m.MinPLR
    model.UCMinProductionLimit = pyo.Constraint(model.rp, model.h, model.hps, rule=uc_min_production_rule)

    if str(parameter["StartUpConstraint"]).strip().lower() == "true":
        Core_model.add_startup_formulation(model)


    # --- Objective ---
    Core_model.add_objective_function(model, True)

    # --- Solve ---
    sum_solver_timer, iterations = Core_model.perform_simple_solve(model, parameter, global_param)
    #sum_solver_timer, iterations = Core_model.perform_simple_solve(model, solver)

    # create a dictionary to store key results
    results_dict = Core_model.create_key_result_dict(model, sum_solver_timer, iterations)

    # add cost shares to the dict from model results
    results_dict["InvestmentCosts"] = sum(
        model.InvestCost * pyo.value(model.hp_invest[hps])
        for hps in model.hps
    ) + sum(
        model.BaseInvestCost * pyo.value(model.hp_base_invest[hps])
        for hps in model.hps
    )

    # --------------------------------------------------
    # Operational electricity costs
    # --------------------------------------------------
    results_dict["ElectricityCosts"] = sum(
        model.ElectricityCost[rp, h]
        * pyo.value(model.p_el[rp, h, hps])
        * model.DeltaH
        * model.rpWeight[rp]
        for rp in model.rp
        for h in model.h
        for hps in model.hps
    )

    # --------------------------------------------------
    # Heat not supplied costs
    # --------------------------------------------------
    results_dict["HeatNotSuppliedCosts"] = sum(
        model.Cost_HeatNotSupplied
        * pyo.value(model.heat_not_supplied[rp, h])
        * model.DeltaH
        * model.rpWeight[rp]
        for rp in model.rp
        for h in model.h
    )

    # --------------------------------------------------
    # Excess heat costs
    # --------------------------------------------------
    results_dict["ExcessHeatCosts"] = sum(
        model.Cost_ExcessHeat
        * pyo.value(model.excess_heat_supplied[rp, h])
        * model.DeltaH
        * model.rpWeight[rp]
        for rp in model.rp
        for h in model.h
    )

    # --------------------------------------------------
    # Startup costs (optional)
    # --------------------------------------------------
    results_dict["StartupCosts"] = sum(
        model.StartupCosts
        * pyo.value(model.startup_hp[rp, h, hps])
        * model.rpWeight[rp]
        for rp in model.rp
        for h in model.h
        for hps in model.hps
    )


    # --------------------------------------------------
    # Total cost (consistency check)
    # --------------------------------------------------
    results_dict["TotalCost_ex_post"] = (
            results_dict["InvestmentCosts"]
            + results_dict["ElectricityCosts"]
            + results_dict["HeatNotSuppliedCosts"]
            + results_dict["ExcessHeatCosts"]
            + results_dict["StartupCosts"]
    )

    #create a dataframe to store time series results
    df_results = Core_model.create_time_series_results(model, df_heat_demand)

    # save the results
    Core_model.save_results(parameter, f"expost_{model_name}", df_results, results_dict)


    return results_dict
