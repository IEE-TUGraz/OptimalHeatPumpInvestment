import pandas as pd
import pyomo.environ as pyo
import os
import data
import Core_model

def solve_UC_model(parameter, df_heat_demand, df_el_price, df_cop_scalor, df_rp_weights, df_warmstart, LP_results):
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

    model.COP = pyo.Param(initialize=3.56)  # fully linearised COP

    Core_model.initialise_investment_bounds(model, parameter, LP_results)

    model.MinPLR = pyo.Param(initialize=parameter["MinPartLoad"])  # minimum part load ratio

    # --- Variables ---
    Core_model.initialise_variables(model)

    # binaries
    Core_model.initialise_binary_variables(model, df_warmstart)


    # --- Model ---
    # add the storage formulation
    Core_model.add_storage_formulation(model)

    # Heat Pump Constraints
    def cop_constraint_rule(m, rp, h, hps):
        return m.q_heat[rp, h, hps] == m.COP * m.p_el[rp, h, hps] * m.COP_Scalor[rp, h]
    model.COPConstraint = pyo.Constraint(model.rp, model.h, model.hps, rule=cop_constraint_rule)

    Core_model.add_investment_formulation(model)

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
    # perform itrative solving if specified
    #sum_solver_time, iterations = Core_model.perform_iterative_solve(model, solver, parameter, global_param)
    #sum_solver_time, iterations = Core_model.perform_simple_solve(model, parameter, global_param)
    sum_solver_time, iterations = Core_model.perform_iterative_solve_2(model, parameter, global_param)

    # create a dictionary to store key results
    results_dict = Core_model.create_key_result_dict(model, sum_solver_time, iterations)


    #create a dataframe to store time series results
    df_results = Core_model.create_time_series_results(model, df_heat_demand)

    # save the results
    Core_model.save_results(parameter, "UC", df_results, results_dict)

    # calculate a unit commitment time series in a separate dataframe
    df_unit_commitment = Core_model.store_warmstart_data(model)

    return results_dict, df_unit_commitment
