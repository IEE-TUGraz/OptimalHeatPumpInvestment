# provide here functions for all core model operations to avoid code duplication
import pyomo.environ as pyo
import pandas as pd
import json
import os


def initialise_sets(model, df_heat_demand):
    model.h = pyo.Set(ordered=True, initialize=df_heat_demand.index.get_level_values('time_h').unique().tolist())
    model.rp = pyo.Set(ordered=True, initialize=df_heat_demand.index.get_level_values('rp').unique().tolist())
    model.hps = pyo.Set(initialize=["HP1", "HP2"])

    return model


def initialise_vector_parameters(model, df_el_price, df_heat_demand, df_cop_scalor, df_rpWeights):
    model.ElectricityCost = pyo.Param(model.rp, model.h, initialize=df_el_price["electricity_price"].to_dict())
    model.HeatDemand = pyo.Param(model.rp, model.h, initialize=df_heat_demand["Q_demand"].to_dict())
    model.COP_Scalor = pyo.Param(model.rp, model.h, initialize=df_cop_scalor["COP_scalor"].to_dict())
    model.rpWeight = pyo.Param(model.rp, initialize=df_rpWeights["weight"].to_dict())

    return model

def initialise_scalar_parameters(model, parameter, global_param):
    time_weights_investment_costs = parameter["DurationDays"] / 365  # hours in a year
    model.DeltaH = pyo.Param(initialize=global_param['DeltaH'])  # in 1/h
    model.InvestCost = pyo.Param(initialize=parameter["InvestmentCost"] * time_weights_investment_costs)  # €/kW/year
    model.BaseInvestCost = pyo.Param(initialize=parameter["BaseInvestmentCost"] *time_weights_investment_costs)  # €/unit
    model.StartupCosts = pyo.Param(initialize=parameter["StartUpCost"])  # €/startup
    model.StorageCap = pyo.Param(initialize=parameter["StorageCapacity"])  # kWh
    model.StorageLoss = pyo.Param(initialize=parameter["StorageSelfDischarge"])  # per hour
    model.StorageChEff = pyo.Param(initialize=parameter["StorageChargeEfficiency"])  # fraction
    model.Cost_ExcessHeat = pyo.Param(initialize=parameter["EHSCost"])  # €/kWh
    model.Cost_HeatNotSupplied = pyo.Param(initialize=parameter["HNSCost"])  # €/kWh

    return model


def initialise_investment_bounds(model, parameter, result_dict=None):
    if result_dict is None:
        model.InvestmentUB = pyo.Param(model.hps, initialize=9999)  # invest_estimate * (1 + rel), mutable=True)  # kW
        model.InvestmentLB = pyo.Param(model.hps, initialize=0)  # invest_estimate / (1 + rel), mutable=True)  # kW
    else:
        rel = parameter["InvBoundRange"]
        invest_estimate = result_dict["HP_Investment_HP1"] + result_dict["HP_Investment_HP2"]
        model.InvestmentUB = pyo.Param(model.hps, initialize={"HP1": invest_estimate*(1+rel), "HP2": invest_estimate/2}, mutable=True) #invest_estimate * (1 + rel), mutable=True)  # kW
        model.InvestmentLB = pyo.Param(model.hps,initialize={"HP1": invest_estimate/2, "HP2": 0}, mutable=True) #invest_estimate / (1 + rel), mutable=True)  # kW


def initialise_variables(model):
    investment_lb_hp1 = pyo.value(model.InvestmentLB["HP1"])
    investment_ub_hp1 = pyo.value(model.InvestmentUB["HP1"])
    investment_lb_hp2 = pyo.value(model.InvestmentLB["HP2"])
    investment_ub_hp2 = pyo.value(model.InvestmentUB["HP2"])

    def hp_invest_bounds(model, hp):
        return (
            pyo.value(model.InvestmentLB[hp]),
            pyo.value(model.InvestmentUB[hp]),
        )
    model.hp_invest = pyo.Var(model.hps, within=pyo.NonNegativeReals,
        initialize=lambda m, hp: (pyo.value(m.InvestmentLB[hp]) + pyo.value(m.InvestmentUB[hp])) / 2,
        bounds=hp_invest_bounds,
    )
    model.p_el = pyo.Var(model.rp, model.h, model.hps, within=pyo.NonNegativeReals)
    model.q_heat = pyo.Var(model.rp, model.h, model.hps, within=pyo.NonNegativeReals)
    model.heat_storage_level = pyo.Var(model.rp, model.h, within=pyo.NonNegativeReals, bounds=(0, model.StorageCap))
    model.heat_storage_charge = pyo.Var(model.rp, model.h, within=pyo.NonNegativeReals)
    model.heat_storage_discharge = pyo.Var(model.rp, model.h, within=pyo.NonNegativeReals)
    model.heat_not_supplied = pyo.Var(model.rp, model.h, within=pyo.NonNegativeReals)
    model.excess_heat_supplied = pyo.Var(model.rp, model.h, within=pyo.NonNegativeReals)
    model.hp_base_invest = pyo.Var(model.hps, within=pyo.Binary, initialize=1)  # heat pump base investment indicator
    return model

def initialise_binary_variables(model, df_warmstart=None):
    if df_warmstart is None:
        model.z_hp = pyo.Var(model.rp, model.h, model.hps, within=pyo.Binary)
        model.startup_hp = pyo.Var(model.rp, model.h, model.hps, within=pyo.Binary)  # heat pump startup indicatorf_warmstart = pd.DataFrame(0, index=pd.MultiIndex.from_product([model.rp, model.h, model.hps], names=["rp", "h", "hps"]), columns=["uc_guess"])
    else:
        model.z_hp = pyo.Var(model.rp, model.h, model.hps, within=pyo.Binary, initialize=df_warmstart["uc_guess"])  # heat pump on/off status
        model.startup_hp = pyo.Var(model.rp, model.h, model.hps, within=pyo.Binary)  # heat pump startup indicator
    return model

def add_storage_formulation(model):
     # Storage Constraints

    def power_balance_rule(m, rp, h):
        return m.HeatDemand[rp, h] == sum(m.q_heat[rp, h, hps] for hps in model.hps) + m.heat_storage_discharge[rp, h] - m.heat_storage_charge[rp, h] + m.heat_not_supplied[rp, h] - m.excess_heat_supplied[rp, h]
    model.PowerBalance = pyo.Constraint(model.rp, model.h, rule=power_balance_rule)

    def heat_storage_balance_rule(m, rp, h):
        if h == m.h.last():
            return pyo.Constraint.Skip

        h_next = m.h.next(h)

        return m.heat_storage_level[rp, h_next] == m.heat_storage_level[rp, h] * (1 - m.StorageLoss * m.DeltaH) + (m.heat_storage_charge[rp, h] * m.StorageChEff - m.heat_storage_discharge[rp, h]) * m.DeltaH
    model.HeatStorageBalance = pyo.Constraint(model.rp, model.h, rule=heat_storage_balance_rule)

    # -------------------------------------------------
    # Cyclic storage constraint
    # -------------------------------------------------
    def heat_storage_cyclic_rule(m, rp):
        h_first = m.h.first()
        h_last = m.h.last()

        return (m.heat_storage_level[rp, h_first] == m.heat_storage_level[rp, h_last] * (1 - m.StorageLoss * m.DeltaH) + (m.heat_storage_charge[rp, h_last] - m.heat_storage_discharge[rp, h_last]) * m.DeltaH)
    model.HeatStorageCyclic = pyo.Constraint(model.rp, rule=heat_storage_cyclic_rule)

    def max_storage_level_rule(m, rp, h):
        return m.heat_storage_level[rp, h] <= m.StorageCap

    model.MaxStorageLevelConstraint = pyo.Constraint(model.rp, model.h, rule=max_storage_level_rule)
    


    return model
    



def add_investment_formulation(model):
    def max_invest_rule(m, rp, h, hps):
        return m.p_el[rp, h, hps] <= m.hp_invest[hps]
    model.UnitCommitmentConstraint = pyo.Constraint(model.rp, model.h, model.hps, rule=max_invest_rule)

    if len(model.hps) == 2:
        def investment_order_rule(model):
            return model.hp_invest["HP1"] >= model.hp_invest["HP2"]
        model.InvestmentOrderConstraint = pyo.Constraint(rule=investment_order_rule)

    def base_investment_rule(m, hps):
        return m.hp_invest[hps] <= m.hp_base_invest[hps] * 1000
    model.BaseInvestmentConstraint = pyo.Constraint(model.hps, rule=base_investment_rule)

    return model


def add_startup_formulation(model):
    def startup_definition_rule(m, rp, h, hps):
        # get previous hour in a cyclic manner
        h_prev = m.h.prev(h) if h != m.h.first() else m.h.last()
        return m.startup_hp[rp, h, hps] >= m.z_hp[rp, h, hps] - m.z_hp[rp, h_prev, hps]

    model.StartupDefinition = pyo.Constraint(model.rp, model.h, model.hps, rule=startup_definition_rule)


def add_objective_function(model, startup_costs_included=False):
    def objective_rule(m):
        operational_costs = sum(m.ElectricityCost[rp, h] * m.p_el[rp, h, hps] * m.DeltaH * m.rpWeight[rp] for rp in model.rp for h in model.h for hps in model.hps)
        investment_costs = sum(m.InvestCost * m.hp_invest[hps] for hps in model.hps) + sum(m.BaseInvestCost * m.hp_base_invest[hps] for hps in model.hps)
        heat_not_supplied_costs = sum(m.Cost_HeatNotSupplied * m.heat_not_supplied[rp, h] * m.DeltaH * m.rpWeight[rp] for rp in model.rp for h in model.h)
        excess_heat_costs = sum(m.Cost_ExcessHeat * m.excess_heat_supplied[rp, h] * m.DeltaH * m.rpWeight[rp] for rp in model.rp for h in model.h)

        if startup_costs_included:
            startup_costs = sum(m.StartupCosts * m.startup_hp[rp, h, hps] * m.rpWeight[rp] for rp in model.rp for h in model.h for hps in model.hps)
            operational_costs += startup_costs

        return operational_costs + investment_costs + heat_not_supplied_costs + excess_heat_costs
    model.TotalCost = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    return model


def create_key_result_dict(model, solver_time, iterations=1):
    results_dict = {
        "TotalCost": pyo.value(model.TotalCost),
        **{f"HP_Investment_{hps}": pyo.value(model.hp_invest[hps]) for hps in model.hps},
        "Heat_Not_Supplied_Total": sum(pyo.value(model.heat_not_supplied[rp, h]) for rp in model.rp for h in model.h),
        "Excess_Heat_Supplied_Total": sum(pyo.value(model.excess_heat_supplied[rp, h]) for rp in model.rp for h in model.h),
        "SolveWork": solver_time,
        "Iterations": iterations
    }
    return results_dict



def create_time_series_results(model, df_heat_demand):
    df_results = pd.DataFrame(index=df_heat_demand.index)
    # create one dataframe column for each heat pump
    for hps in model.hps:
        df_results[f"Electricity_Consumption_kW_{hps}"] = [pyo.value(model.p_el[rp, h, hps]) for rp in model.rp for h in model.h]
        df_results[f"Heat_Produced_kW_{hps}"] = [pyo.value(model.q_heat[rp, h, hps]) for rp in model.rp for h in model.h]
        df_results["Heat_Storage_Level_kWh"] = [pyo.value(model.heat_storage_level[rp, h]) for rp in model.rp for h in model.h]
        df_results["Heat_Storage_Charge_kW"] = [pyo.value(model.heat_storage_charge[rp, h]) for rp in model.rp for h in model.h]
        df_results["Heat_Storage_Discharge_kW"] = [pyo.value(model.heat_storage_discharge[rp, h]) for rp in model.rp for h in model.h]
        df_results["Heat_Not_Supplied_kW"] = [pyo.value(model.heat_not_supplied[rp, h]) for rp in model.rp for h in model.h]
        df_results["Excess_Heat_Supplied_kW"] = [pyo.value(model.excess_heat_supplied[rp, h]) for rp in model.rp for h in model.h]
        df_results["HeatDemand"] = [pyo.value(model.HeatDemand[rp, h]) for rp in model.rp for h in model.h]
    return df_results


def store_warmstart_data(model):
    idx_rp_h_hps = pd.MultiIndex.from_product([model.rp, model.h, model.hps], names=["rp", "h", "hps"])

    df_warmstart = pd.DataFrame(index=idx_rp_h_hps)

    df_warmstart["uc_guess"] = [
        pyo.value(model.z_hp[rp, h, hps])
        for rp in model.rp
        for h in model.h
        for hps in model.hps
    ]
    return df_warmstart


def initialise_piecewise_linear_parameters(model):
    model.z_PL = pyo.Var(model.rp, model.h, model.s, model.hps, within=pyo.Binary)  # segment selection variable
    model.d_scaled_pos = pyo.Var(model.rp, model.h, model.s, model.hps, within=pyo.NonNegativeReals)
    model.d_scaled_neg = pyo.Var(model.rp, model.h, model.s, model.hps, within=pyo.NonNegativeReals)
    # 0.25 PLR - not updated yet
    #model.K = pyo.Param(model.s, initialize={'s1': 5.819548872180451, 's2': 5.348837209302325, 's3': 4.767441860465116, 's4': 3.953488372093023})  # segment slope
    #model.D_pos = pyo.Param(model.s, initialize={'s1': 0, 's2': 0, 's3': 0, 's4': 0.062935905633207})
    #model.D_neg = pyo.Param(model.s, initialize={'s1': 0.4646009714531245, 's2': 0.42549508089773996, 's3': 0.26669910429670696, 's4': 0})
    #model.RMin = pyo.Param(model.s, initialize={'s1': 0.55, 's2': 0.6625000000000001, 's3': 0.775, 's4': 0.8875})  # segment minimum PLR
    #model.RMax = pyo.Param(model.s, initialize={'s1': 0.6625000000000001, 's2': 0.775, 's3': 0.8875, 's4': 1.0})  # segment maximum PLR

    model.K = pyo.Param(model.s, initialize={'s1': 3.6685, 's2': 3.6800000000000037, 's3': 3.6800000000000037, 's4': 3.648000000000001})  # segment slope
    model.D_pos = pyo.Param(model.s, initialize={'s1': 0, 's2': 0, 's3': 0, 's4': 0})
    model.D_neg = pyo.Param(model.s, initialize={'s1': 0.11653333333333327, 's2': 0.11200000000000195, 's3': 0.11340000000000242, 's4': 0.08746666666666728})
    model.RMin = pyo.Param(model.s, initialize={'s1': 0.1, 's2': 0.325, 's3': 0.55, 's4': 0.775})  # segment minimum PLR
    model.RMax = pyo.Param(model.s, initialize={'s1': 0.325, 's2': 0.55, 's3': 0.775, 's4': 1.0})  # segment maximum PLR

    # 0.55 PLR
    #model.K = pyo.Param(model.s, initialize={'s1': 4.273207427813483, 's2': 4.215739077645132, 's3': 4.011641097847147, 's4': 3.640790562187536})  # segment slope
    #model.D_pos = pyo.Param(model.s, initialize={'s1': 0, 's2': 0, 's3': 0, 's4': 0.062935905633207})
    #model.D_neg = pyo.Param(model.s, initialize={'s1': 0.4646009714531245, 's2': 0.42549508089773996, 's3': 0.26669910429670696, 's4': 0})
    #model.RMin = pyo.Param(model.s, initialize={'s1': 0.55, 's2': 0.6625000000000001, 's3': 0.775, 's4': 0.8875})  # segment minimum PLR
    #model.RMax = pyo.Param(model.s, initialize={'s1': 0.6625000000000001, 's2': 0.775, 's3': 0.8875, 's4': 1.0})  # segment maximum PLR
    return model

def add_linear_underestimator(model, K_lu, D_lu, D_offset=0.0):
    model.K_lu  = pyo.Param(initialize=K_lu)  # slope of linear underestimator
    model.D_lu = pyo.Param(initialize=D_lu + D_offset)  # intercept of linear underestimator + ofset for saftexy margin

    def linear_underestimator_rule(m, rp, h, hps):
        return m.q_heat[rp, h, hps] >= (m.K_lu * m.p_el[rp, h, hps] - m.D_lu * m.hp_invest[hps]) * m.COP_Scalor[rp, h]
    model.LinearUnderestimatorConstraint = pyo.Constraint(model.rp, model.h, model.hps, rule=linear_underestimator_rule)


    print("Linear underestimator added to the model.")

    return model



def save_results(parameter, name_prefix: str, df_results, key_results):
    import os

    if not os.path.exists("results"):
        os.makedirs("results")

    scenario_folder = os.path.join("results", "scenario_" + str(parameter['ScenarioIndex']))
    if not os.path.exists(scenario_folder):
        os.makedirs(scenario_folder)

    df_results.to_excel(os.path.join(scenario_folder, name_prefix + "_results.xlsx"))

    # save key results as well
    with open(os.path.join(scenario_folder, name_prefix + "_key_results.json"), "w") as f:
        json.dump(key_results, f, indent=4)

    return


def perform_simple_solve(model, parameter, global_param):
    solver = pyo.SolverFactory('gurobi_persistent')
    solver.set_instance(model)
    solver.options["MIPGap"] = 0.01
    #solver.options["TimeLimit"] = global_param["solver_time_limit"]
    results = solver.solve(model, tee=False, warmstart=True)
    solve_work = solver._solver_model.Work
    return solve_work, 1


def perform_iterative_solve(model, solver, parameter, global_param):
    max_iterations = global_param['max_iterations']
    overlap_tol = global_param['overlap_tolerance']
    rel = parameter["InvBoundRange"]
    sum_solver_time = 0

    # --- Iterative refinement loop ---
    for it in range(max_iterations):

        results = solver.solve(model, tee=True, warmstart=True)
        hp = pyo.value(model.hp_invest)
        sum_solver_time += results.solver.time

        lb = pyo.value(model.InvestmentLB)
        ub = pyo.value(model.InvestmentUB)

        print(f"[Iter {it}] hp_invest = {hp:.6f}, bounds = [{lb:.6f}, {ub:.6f}]")

        lower_active = hp <= lb + float(global_param["constraint_binding_tolerance"])
        upper_active = hp >= ub - float(global_param["constraint_binding_tolerance"])

        # --- Case 1: neither bound is active → stop here ---
        if not lower_active and not upper_active:
            print("No investment bound active — stopping.")
            break

        # --- Case 2: Lower bound active ---
        if lower_active:
            print(f"⚠ Lower bound active in scenario {parameter['ScenarioIndex']}")

            # Relax lower bound
            new_lb = lb * (1 - rel)
            new_lb = max(new_lb, 0)

            model.hp_invest.setlb(new_lb)

            # Reset upper bound to initial tightness + overlap tolerance
            new_ub = lb + overlap_tol
            model.hp_invest.setub(new_ub)

            print(f" → New LB={new_lb:.3f}, UB reset to {new_ub:.3f}")

        # --- Case 3: Upper bound active ---
        if upper_active:
            print(f"⚠ Upper bound active in scenario {parameter['ScenarioIndex']}")

            # Relax upper bound
            new_ub = ub / (1 - rel) + overlap_tol

            model.hp_invest.setub(new_ub)

            # Reset lower bound to initial tightness + overlap tolerance
            new_lb = ub - overlap_tol
            model.hp_invest.setlb(new_lb)

            print(f" → New UB={new_ub:.3f}, LB reset to {new_lb:.3f}")

        # set the new bounds to the model parameter
        model.InvestmentLB.set_value(new_lb)
        model.InvestmentUB.set_value(new_ub)

    else:
        print("⚠ Maximum iterations reached. Bound still active.")

    return sum_solver_time, it + 1


def perform_iterative_solve_2(model, parameter, global_param):
    max_iterations = global_param["max_iterations"]
    overlap_tol = global_param["overlap_tolerance"]
    bind_tol = global_param["constraint_binding_tolerance"]
    rel = parameter["InvBoundRange"]
    solver = pyo.SolverFactory('gurobi_persistent')
    solver.set_instance(model)
    solver.options["MIPGap"] = parameter["MIPGap"]
    solver.options["TimeLimit"] = global_param["solver_time_limit"]

    sum_work_time = 0

    # --- Iterative refinement loop ---
    for it in range(max_iterations):
        results = solver.solve(tee=True, warmstart=True)
        sum_work_time = solver._solver_model.Work


        any_bound_active = False

        print(f"\n[Iter {it}]")

        # --- Loop over heat pumps ---
        for hp in model.hps:
            hp_val = pyo.value(model.hp_invest[hp])
            lb = pyo.value(model.InvestmentLB[hp])
            ub = pyo.value(model.InvestmentUB[hp])

            print(
                f"  {hp}: hp_invest={hp_val:.6f}, "
                f"bounds=[{lb:.6f}, {ub:.6f}]"
            )

            lower_active = (hp_val <= lb + bind_tol) & (hp_val != 0)
            upper_active = hp_val >= ub - bind_tol

            # --- If neither bound is active → nothing to do for this HP ---
            if not lower_active and not upper_active:
                continue

            any_bound_active = True

            # --- Case 1: Lower bound active ---
            if lower_active:
                print(f"  ⚠ {hp}: lower bound active")

                new_lb = max(lb * (1 - rel), 0)
                new_ub = lb + overlap_tol

            # --- Case 2: Upper bound active ---
            elif upper_active:
                print(f"  ⚠ {hp}: upper bound active")

                new_ub = ub / (1 - rel) + overlap_tol
                new_lb = ub - overlap_tol

            # --- Apply bounds to variable ---
            model.hp_invest[hp].setlb(new_lb)
            model.hp_invest[hp].setub(new_ub)

            # --- Store bounds back to Params ---
            model.InvestmentLB[hp].set_value(new_lb)
            model.InvestmentUB[hp].set_value(new_ub)

            print(
                f"    → New bounds for {hp}: "
                f"LB={new_lb:.3f}, UB={new_ub:.3f}"
            )

        # --- Stop if no HP had an active bound ---
        if not any_bound_active:
            print("No investment bounds active — stopping.")
            break

    else:
        print("⚠ Maximum iterations reached. Some bounds still active.")

    return sum_work_time, it + 1

