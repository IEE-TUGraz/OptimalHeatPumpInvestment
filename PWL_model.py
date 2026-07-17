import pandas as pd
import pyomo.environ as pyo
import os
import data
import Core_model
import segment_fit

def solve_PWL_model(parameter, df_heat_demand, df_el_price, df_cop_scalor, df_rp_weights, df_warmstart, LP_results):
    global_param = data.load_parameter()
    model = pyo.ConcreteModel()

    rel = parameter["InvBoundRange"]

    # --- Sets ---
    Core_model.initialise_sets(model, df_heat_demand)

    # --- Parameters ---
    # vectors
    Core_model.initialise_vector_parameters(model, df_el_price, df_heat_demand, df_cop_scalor, df_rp_weights)

    # scalars
    Core_model.initialise_scalar_parameters(model, parameter, global_param)
    Core_model.initialise_investment_bounds(model, parameter, LP_results) #LP_results["HP_Investment"])

    model.MinPLR = pyo.Param(initialize=parameter["MinPartLoad"])  # minimum part load ratio

    # piecewise linear COP: secant fit with n_fit_segments segments (scenario param, default 4)
    n_seg = segment_fit.resolve_n_segments(parameter, global_param)
    fit = segment_fit.fit_cop_segments(n_seg)
    print(f"[PWL] piecewise-linear COP fit: {n_seg} secant segment(s).")
    Core_model.initialise_piecewise_linear_parameters(model, fit)

    # --- Variables ---
    Core_model.initialise_variables(model)
    

    # for piecewise linear COP
    model.p_elec_seg = pyo.Var(model.rp, model.h, model.s, model.hps, within=pyo.NonNegativeReals)  # electricity consumption per segment

    # binaries
    Core_model.initialise_binary_variables(model, df_warmstart)

    # --- Model ---
    # add the storage formulation
    Core_model.add_storage_formulation(model)

    # firm supply on design/extreme periods (HNS=0); penalized slack unchanged on normal periods
    Core_model.enforce_firm_design_supply(model)

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
    sum_solver_timer, iterations = Core_model.perform_iterative_solve_2(model, parameter, global_param)
    #sum_solver_timer, iterations = Core_model.perform_simple_solve(model, solver)

    # create a dictionary to store key results
    results_dict = Core_model.create_key_result_dict(model, sum_solver_timer, iterations)

    #create a dataframe to store time series results
    df_results = Core_model.create_time_series_results(model, df_heat_demand)

    # save the results
    Core_model.save_results(parameter, "PWL", df_results, results_dict)


    return results_dict
