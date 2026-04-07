import pandas as pd
import pyomo.environ as pyo
import os
import data
import Core_model

def solve_CR_model(parameter, df_heat_demand, df_el_price, df_cop_scalor, df_rp_weight, df_warmstart, LP_results):
    global_param = data.load_parameter()
    model = pyo.ConcreteModel()


    # --- Sets ---
    Core_model.initialise_sets(model, df_heat_demand)

    # --- Parameters ---
    # vectors
    Core_model.initialise_vector_parameters(model, df_el_price, df_heat_demand, df_cop_scalor, df_rp_weight)

    # scalars
    Core_model.initialise_scalar_parameters(model, parameter, global_param)
    Core_model.initialise_investment_bounds(model, parameter, LP_results)

    model.MinPLR = pyo.Param(initialize=parameter["MinPartLoad"])  # minimum part load ratio

    # paper model
    #model.A = pyo.Param(initialize=-0.94843333)  # a = -0.8905
    #model.B = pyo.Param(initialize=5.54024847)#4.694)  # b
    #model.C = pyo.Param(initialize=0.88370944) #0.7996)  # c # opposite sign!!!
    # datashee model
    model.A = pyo.Param(initialize=-0.03090909)  # a = -0.8905
    model.B = pyo.Param(initialize=3.71587273)#4.694)  # b
    model.C = pyo.Param(initialize=0.12344) #0.7996)  # c # opposite sign!!!

    # --- Variables ---
    Core_model.initialise_variables(model)
    investment_ub = pyo.value(model.InvestmentUB["HP1"])
    model.r_a = pyo.Var(model.rp, model.h, model.hps, within=pyo.NonNegativeReals, bounds=(0, investment_ub / 2))  # auxiliary variable for the conic constraint
    model.c_scaled = pyo.Var(model.rp, model.h, model.hps, within=pyo.NonNegativeReals, bounds=(0, model.C * investment_ub))  # now implicitly restricted to positive values and then substracted (see below)

    # binaries
    Core_model.initialise_binary_variables(model, df_warmstart)

    # --- Model ---
    # add the storage formulation
    Core_model.add_storage_formulation(model)


    # Heat Pump Constraints
    def uc_cop_constraint_rule(m, rp, h, hps):
        return m.q_heat[rp, h, hps] == (2 * m.A * m.r_a[rp, h, hps] + m.B * m.p_el[rp, h, hps] - m.c_scaled[rp, h, hps]) * m.COP_Scalor[rp, h]  # it seems more performant to have it as equality; but check again
    model.COPUCConstraint = pyo.Constraint(model.rp, model.h, model.hps, rule=uc_cop_constraint_rule)

    def A_relaxation_rule(m, rp, h, hps):
        return m.p_el[rp, h, hps]**2 <= 2 * m.r_a[rp, h, hps] * m.hp_invest[hps]
    model.ARelationConstraint = pyo.Constraint(model.rp, model.h, model.hps, rule=A_relaxation_rule)

    def intercept_active_segment_rule(m, rp, h, hps):
        return m.c_scaled[rp, h, hps] <= m.z_hp[rp, h, hps] * m.C * m.InvestmentUB[hps]
    model.InterceptActiveSegmentConstraint = pyo.Constraint(model.rp, model.h, model.hps, rule=intercept_active_segment_rule)

    def intercept_upper_bound_rule(m, rp, h, hps):
        return m.c_scaled[rp, h, hps] <= m.C * m.hp_invest[hps]
    model.InterceptUpperBoundConstraint = pyo.Constraint(model.rp, model.h, model.hps, rule=intercept_upper_bound_rule)

    def intercept_lower_bound_rule(m, rp, h, hps):
        return m.c_scaled[rp, h, hps] >= m.C * m.hp_invest[hps] - (1 - m.z_hp[rp, h, hps]) * m.InvestmentUB[hps] * m.C
    model.InterceptLowerBoundConstraint = pyo.Constraint(model.rp, model.h, model.hps, rule=intercept_lower_bound_rule)

    Core_model.add_investment_formulation(model)

    def uc_max_production_rule(m, rp, h, hps):
        return m.p_el[rp, h, hps] <= m.z_hp[rp, h, hps] * m.InvestmentUB[hps]
    model.UCProductionLimit = pyo.Constraint(model.rp, model.h, model.hps, rule=uc_max_production_rule)

    def uc_min_production_rule(m, rp, h, hps):
        return m.p_el[rp, h, hps] >= m.hp_invest[hps] * m.MinPLR - (1 - m.z_hp[rp, h, hps]) * m.InvestmentUB[hps] * m.MinPLR
    model.UCMinProductionLimit = pyo.Constraint(model.rp, model.h, model.hps, rule=uc_min_production_rule)

    if str(parameter["StartUpConstraint"]).strip().lower() == "true":
        Core_model.add_startup_formulation(model)


    if str(parameter["LinearUnderestimator"]).strip().lower() == "true":
        Core_model.add_linear_underestimator(model, K_lu=3.6818727272727254, D_lu=0.1203490909090914, D_offset=0.00)  # for datasheet model

    # --- Objective ---
    Core_model.add_objective_function(model, True)

    # --- Solve ---
    # iterative solving
    #sum_solver_time, it = Core_model.perform_iterative_solve(model, solver, parameter, global_param)
    sum_solver_time, it = Core_model.perform_iterative_solve_2(model, parameter, global_param)

    # simple solve
    #sum_solver_time, it = Core_model.perform_simple_solve(model, solver)

    # create a dictionary to store key results
    results_dict = Core_model.create_key_result_dict(model, sum_solver_time, it)

    #create a dataframe to store time series results
    df_results = Core_model.create_time_series_results(model, df_heat_demand)

    # check of there exists a folder "results", if not create it
    Core_model.save_results(parameter, "CR", df_results, results_dict)

    # calculate a unit commitment time series in a separate dataframe
    df_unit_commitment = Core_model.store_warmstart_data(model)

    return results_dict
