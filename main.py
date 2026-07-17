import os
import pandas as pd
import sys
from termcolor import cprint


def _flush_excel(df, path):
    """Atomically write a roll-up DataFrame (temp file + os.replace) so a crash or
    kill mid-write can never corrupt or truncate the existing results file."""
    tmp = path + ".tmp.xlsx"
    df.to_excel(tmp, index=False)
    os.replace(tmp, path)

import data
import LP_model
import UC_model
import PWL_model
import PWLR_model
import CR_model
import expost_analysis
import expost_model
import expost_out_of_sample

import TimeSeriesAggregation


# NOTE: the __main__ guard is required -- the out-of-sample ex-post runs its formulations in parallel
# processes (ProcessPoolExecutor); on Windows those child processes re-import this module, so the
# pipeline must not execute at import time or it would spawn recursively.
if __name__ == "__main__":

    scenarios = data.load_input_scenarios("inputScenarios.xlsx")
    model_data = data.load_data("IBK_case_study_data.pkl")
    global_params = data.load_parameter()

    scenario_results = scenarios.copy()

    all_KPI_results = []
    all_OOS_KPI_results = []

    # iterate through scenarios and print each one
    for index, scenario in scenarios.iterrows():
        print(f"Start processing scenario {scenario['ScenarioIndex']} with {scenario['DurationDays']*24*4} time steps.", "cyan")
        print(f"Current scenario parameters are: {scenario}")

        # create a dictionary of parameters out of the scenario row
        scenario_params = scenario.to_dict()

        # prepare data for the current scenario
        df_heat_demand_full, df_el_price_full, df_cop_scalor_full = data.extract_relevant_data(model_data, scenario_params)

        # perform time series aggregation if specified in the scenario
        df_heat_demand, df_el_price, df_cop_scalor, df_rpWeights = TimeSeriesAggregation.performe_TSA(
            df_heat_demand_full, df_el_price_full, df_cop_scalor_full,
            global_params["num_rep_periods"], global_params["hours_per_period"],
            storage_capacity_kwh=scenario_params["StorageCapacity"],
            storage_ch_eff=scenario_params["StorageChargeEfficiency"],
            storage_loss=scenario_params["StorageSelfDischarge"],
            resolution_h=global_params["DeltaH"],
        )
        print(f"Time series aggregation performed. Reduced to {df_heat_demand.index.get_level_values('rp').nunique()} representative periods of tbd hours each.\n")

        # save the representative period weights for later use
        data.save_rp_weights(df_rpWeights, scenario_params)

        # save data for ex-post analysis
        data.save_data_to_disk(df_el_price, df_rpWeights, df_heat_demand, df_cop_scalor, scenario)

        # solve the lp model in a completly chronological
        df_heat_demand_chrono, df_el_price_chrono, df_cop_scalor_chrono, df_rpWeights_chrono = TimeSeriesAggregation.adjust_format_chrono(df_heat_demand_full, df_el_price_full, df_cop_scalor_full)

        LP_chrono_keyResults, _ = LP_model.solve_LP_model(scenario_params, df_heat_demand_chrono, df_el_price_chrono, df_cop_scalor_chrono, df_rpWeights_chrono)
        print(f"Key results LP Chrono: {LP_chrono_keyResults}\n")

        # solve the scenario as linear program
        LP_keyResults, df_LP_unit_commitment = LP_model.solve_LP_model(scenario_params, df_heat_demand, df_el_price, df_cop_scalor, df_rpWeights)
        cprint(f"Finished processing scenario {scenario['ScenarioIndex']} as LP.\n", "magenta")
        print(f"Key results LP: {LP_keyResults}\n")

        # calculate the difference in costs and investment between chronological and TSA LP model
        cost_diff = LP_keyResults["TotalCost"] - LP_chrono_keyResults["TotalCost"]
        #invest_diff = LP_keyResults["HP_Investment"] - LP_chrono_keyResults["HP_Investment"]
        print(f"Difference in Total Cost between TSA LP and Chronological LP: {cost_diff:.2f}")
        #print(f"Difference in HP Investment between TSA LP and Chronological LP: {invest_diff:.3f}\n")

        # write key results back to the scenario_results dataframe and add a prefix "LP" to the column names
        for key, value in LP_keyResults.items():
            scenario_results.at[index, "LP_" + key] = value

        # solve the scenario as mixed-integer program with unit commitment
        UC_keyResults, df_UC_unit_commitment = UC_model.solve_UC_model(scenario_params, df_heat_demand, df_el_price, df_cop_scalor, df_rpWeights, df_LP_unit_commitment, LP_keyResults)
        cprint(f"Finished processing scenario {scenario['ScenarioIndex']} as UC.\n", "magenta")
        print(f"Key results UC: {UC_keyResults}\n")
        # write key results back to the scenario_results dataframe and add a prefix "UC" to the column names
        for key, value in UC_keyResults.items():
            scenario_results.at[index, "UC_" + key] = value

        # solve as piecewise linear model
        PWL_keyResults = PWL_model.solve_PWL_model(scenario_params, df_heat_demand, df_el_price, df_cop_scalor, df_rpWeights, df_UC_unit_commitment, UC_keyResults)
        cprint(f"Finished processing scenario {scenario['ScenarioIndex']} as PWL.\n", "magenta")
        print(f"Key results PWL: {PWL_keyResults}\n")

        # write key results back to the scenario_results dataframe and add a prefix "PWL" to the column names
        for key, value in PWL_keyResults.items():
            scenario_results.at[index, "PWL_" + key] = value

        # solve as releaxed piecewise linear model
        PWLR_keyResults, df_PWLR_unit_commitment = PWLR_model.solve_PWLR_model(scenario_params, df_heat_demand, df_el_price, df_cop_scalor, df_rpWeights, df_UC_unit_commitment, UC_keyResults)
        cprint(f"Finished processing scenario {scenario['ScenarioIndex']} as PWLR.\n", "magenta")
        print(f"Key results PWLR: {PWLR_keyResults}\n")

        # write key results back to the scenario_results dataframe and add a prefix "PWLR" to the column names
        for key, value in PWLR_keyResults.items():
            scenario_results.at[index, "PWLR_" + key] = value


        # solve as convex relaxation model
        CR_keyResults = CR_model.solve_CR_model(scenario_params, df_heat_demand, df_el_price, df_cop_scalor, df_rpWeights, df_UC_unit_commitment, UC_keyResults)
        cprint(f"Finished processing scenario {scenario['ScenarioIndex']} as CR.\n", "magenta")
        print(f"Key results CR: {CR_keyResults}\n")

        # write key results back to the scenario_results dataframe and add a prefix "CR" to the column names
        for key, value in CR_keyResults.items():
            scenario_results.at[index, "CR_" + key] = value


        # in-sample ex-post: validate each formulation on the aggregated representative periods
        KPI_expost = expost_analysis.run_kpi_expost_analysis(scenario["ScenarioIndex"], scenario_params)
        all_KPI_results.append(KPI_expost)

        # Flush the investment + in-sample roll-ups NOW, before the long/fragile OOS step, so an OOS
        # failure (or an interrupted run) can never lose the aggregates that are already complete.
        _flush_excel(scenario_results, "results/Scenario_Results.xlsx")
        _flush_excel(pd.concat(all_KPI_results, ignore_index=True), "results/All_KPI_Results.xlsx")

        # out-of-sample ex-post: re-solve each formulation's fixed investment on the FULL-resolution
        # chronological series (rolling horizon, formulations run in parallel processes). Toggled/tuned
        # via parameter.yaml (run_oos_analysis, oos_chunk_steps, oos_overlap_steps, oos_mipgap, oos_n_jobs).
        if global_params.get("run_oos_analysis", True):
            OOS_KPI = expost_out_of_sample.run_oos_expost_analysis(
                scenario["ScenarioIndex"], scenario_params, model_data,
                chunk_steps=global_params.get("oos_chunk_steps", 672),
                overlap_steps=global_params.get("oos_overlap_steps", 96),
                mipgap=global_params.get("oos_mipgap", 1e-4),
                time_limit=global_params.get("oos_chunk_time_limit", 900),
                n_jobs=global_params.get("oos_n_jobs", None),
            )
            all_OOS_KPI_results.append(OOS_KPI)
            _flush_excel(pd.concat(all_OOS_KPI_results, ignore_index=True), "results/All_KPI_Results_OOS.xlsx")

        # plot all performance maps for the scenario
        expost_analysis.plot_all_performance_maps(scenario["ScenarioIndex"])

        # plot all time plots
        expost_analysis.plot_all_models(scenario["ScenarioIndex"], scenario_params["StorageCapacity"])

    # final authoritative roll-up write (the per-scenario flushes above already keep these current)
    _flush_excel(scenario_results, "results/Scenario_Results.xlsx")
    _flush_excel(pd.concat(all_KPI_results, ignore_index=True), "results/All_KPI_Results.xlsx")

    # save the aggregated out-of-sample KPIs (only if the OOS step ran)
    if all_OOS_KPI_results:
        _flush_excel(pd.concat(all_OOS_KPI_results, ignore_index=True), "results/All_KPI_Results_OOS.xlsx")
