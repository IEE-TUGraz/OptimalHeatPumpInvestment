import pandas as pd
import pyomo.environ as pyo
import os
import data
import Core_model

def solve_LP_model(parameter, df_heat_demand, df_el_price, df_cop_scalor, df_rpWeights):
    global_param = data.load_parameter()

    model = pyo.ConcreteModel()

    # --- Sets ---
    Core_model.initialise_sets(model, df_heat_demand)

    # --- Parameters ---
    # vectors
    Core_model.initialise_vector_parameters(model, df_el_price, df_heat_demand, df_cop_scalor, df_rpWeights)

    # scalars
    Core_model.initialise_scalar_parameters(model, parameter, global_param)
    model.COP = pyo.Param(initialize=3.56) # fully linearised COP

    Core_model.initialise_investment_bounds(model, parameter)

    # --- Variables ---
    Core_model.initialise_variables(model)

    # --- Model ---
    Core_model.add_storage_formulation(model)

    # firm supply on design/extreme periods (HNS=0); penalized slack unchanged on normal periods
    Core_model.enforce_firm_design_supply(model)


    # Heat Pump Constraints
    def cop_constraint_rule(m, rp, h, hps):
        return m.q_heat[rp, h, hps] == m.COP * m.p_el[rp, h, hps] * m.COP_Scalor[rp, h]
    model.COPConstraint = pyo.Constraint(model.rp, model.h, model.hps, rule=cop_constraint_rule)

    Core_model.add_investment_formulation(model)

    # --- Objective ---
    Core_model.add_objective_function(model, False)

    # --- Solve ---
    solver = pyo.SolverFactory('gurobi_persistent')
    solver.set_instance(model)
    #solver.options["MIPGap"] = parameter["MIPGap"]
    results = solver.solve(tee=True)
    work_time = solver._solver_model.Work

    # create a dictionary to store key results
    results_dict = Core_model.create_key_result_dict(model, work_time)

    # store time sereies results in a dataframe
    df_results = Core_model.create_time_series_results(model, df_heat_demand)


    # save the results
    Core_model.save_results(parameter, "LP", df_results, results_dict)

    # calculate a unit commitment time series in a separate dataframe
    idx_rp_h_hps = pd.MultiIndex.from_product([model.rp, model.h, model.hps], names=["rp", "h", "hps"])

    df_unit_commitment = pd.DataFrame(index=idx_rp_h_hps)

    df_unit_commitment["uc_guess"] = [
        1 if pyo.value(model.p_el[rp, h, hps]) > 0.10 else 0
        for rp in model.rp
        for h in model.h
        for hps in model.hps
    ]

    return results_dict, df_unit_commitment
