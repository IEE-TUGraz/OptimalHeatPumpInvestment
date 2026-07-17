import pandas as pd
import pyomo.environ as pyo
import os
import data
import Core_model
import segment_fit

def solve_PWLR_model(parameter, df_heat_demand, df_el_price, df_cop_scalor, df_rp_weight, df_warmstart, LP_results):
    global_param = data.load_parameter()

    model = pyo.ConcreteModel()

    rel = parameter["InvBoundRange"]

    # --- Sets ---
    Core_model.initialise_sets(model, df_heat_demand)

    # --- Parameters ---
    # vectors
    Core_model.initialise_vector_parameters(model, df_el_price, df_heat_demand, df_cop_scalor, df_rp_weight)

    # scalars
    Core_model.initialise_scalar_parameters(model, parameter, global_param)
    Core_model.initialise_investment_bounds(model, parameter, LP_results)

    model.MinPLR = pyo.Param(initialize=parameter["MinPartLoad"])  # minimum part load ratio

    # piecewise linear COP: secant fit with n_fit_segments segments (scenario param, default 4)
    n_seg = segment_fit.resolve_n_segments(parameter, global_param)
    fit = segment_fit.fit_cop_segments(n_seg)
    print(f"[PWLR] piecewise-linear COP fit: {n_seg} secant segment(s).")
    Core_model.initialise_piecewise_linear_parameters(model, fit)

    # --- Variables ---
    Core_model.initialise_variables(model)

    # binaries
    Core_model.initialise_binary_variables(model, df_warmstart)

    # --- Model ---
    # add the storage formulation
    Core_model.add_storage_formulation(model)

    # firm supply on design/extreme periods (HNS=0); penalized slack unchanged on normal periods
    Core_model.enforce_firm_design_supply(model)

    # Heat Pump Constraints
    def uc_cop_constraint_rule(m, rp, h, s, hps):
        return m.q_heat[rp, h, hps] <= (m.K[s] * m.p_el[rp, h, hps] + m.d_scaled_pos[rp, h, s, hps] - m.d_scaled_neg[rp, h, s, hps]) * m.COP_Scalor[rp, h]
    model.COPUCConstraint = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=uc_cop_constraint_rule)

    def intercept_active_segment_rule_pos(m, rp, h, s, hps):
        return m.d_scaled_pos[rp, h,s, hps] <= m.z_hp[rp, h, hps] * m.D_pos[s] * m.InvestmentUB[hps]
    model.InterceptActiveSegmentConstraint_pos = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=intercept_active_segment_rule_pos)

    def intercept_upper_bound_rule_pos(m, rp, h, s, hps):
        return m.d_scaled_pos[rp, h,s, hps] <= m.D_pos[s] * m.hp_invest[hps]
    model.InterceptUpperBoundConstraint_pos = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=intercept_upper_bound_rule_pos)

    def intercept_lower_bound_rule_pos(m, rp, h, s, hps):
        return m.d_scaled_pos[rp, h,s, hps] >= m.D_pos[s] * m.hp_invest[hps] - (1 - m.z_hp[rp, h, hps]) * m.InvestmentUB[hps] * m.D_pos[s]
    model.InterceptLowerBoundConstraint_pos = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=intercept_lower_bound_rule_pos)

    def intercept_active_segment_rule_neg(m, rp, h, s, hps):
        return m.d_scaled_neg[rp, h,s, hps] <= m.z_hp[rp, h, hps] * m.D_neg[s] * m.InvestmentUB[hps]
    model.InterceptActiveSegmentConstraint_neg = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=intercept_active_segment_rule_neg)

    def intercept_upper_bound_rule_neg(m, rp, h, s, hps):
        return m.d_scaled_neg[rp, h, s, hps] <= m.D_neg[s] * m.hp_invest[hps]
    model.InterceptUpperBoundConstraint_neg = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=intercept_upper_bound_rule_neg)

    def intercept_lower_bound_rule_neg(m, rp, h, s, hps):
        return m.d_scaled_neg[rp, h, s, hps] >= m.D_neg[s] * m.hp_invest[hps] - (1 - m.z_hp[rp, h, hps]) * m.InvestmentUB[hps] * m.D_neg[s]
    model.InterceptLowerBoundConstraint_neg = pyo.Constraint(model.rp, model.h, model.s, model.hps, rule=intercept_lower_bound_rule_neg)

    def uc_max_production_rule(m, rp, h, hps):
        return m.p_el[rp, h, hps] <= m.z_hp[rp, h, hps] * m.InvestmentUB[hps]
    model.UCProductionLimit = pyo.Constraint(model.rp, model.h, model.hps, rule=uc_max_production_rule)

    def uc_min_production_rule(m, rp, h, hps):
        return m.p_el[rp, h, hps] >= m.hp_invest[hps] * m.MinPLR - (1 - m.z_hp[rp, h, hps]) * m.InvestmentUB[hps] * m.MinPLR
    model.UCMinProductionLimit = pyo.Constraint(model.rp, model.h, model.hps, rule=uc_min_production_rule)

    if str(parameter["StartUpConstraint"]).strip().lower() == "true":
        Core_model.add_startup_formulation(model)

    if str(parameter["LinearUnderestimator"]).strip().lower() == "true":
        # tightest global underestimator of the same n-segment secant fit (kept consistent with n_fit_segments)
        Core_model.add_linear_underestimator(model, K_lu=fit["K_lu"], D_lu=fit["D_lu"], D_offset=0.00)

    Core_model.add_investment_formulation(model)

    # --- Objective ---
    Core_model.add_objective_function(model, True)

    # --- Solve ---
    # perorm itrative solving
    sum_solver_time, iterations = Core_model.perform_iterative_solve_2(model, parameter, global_param)
    #sum_solver_time, iterations = Core_model.perform_simple_solve(model, solver)

    # create a dictionary to store key results
    results_dict = Core_model.create_key_result_dict(model, sum_solver_time, iterations)

    #create a dataframe to store time series results
    df_results = Core_model.create_time_series_results(model, df_heat_demand)

    # save the results
    Core_model.save_results(parameter, "PWLR", df_results, results_dict)


    # get the unit commitment decision time series
    df_unit_commitment = Core_model.store_warmstart_data(model)

    return results_dict, df_unit_commitment
