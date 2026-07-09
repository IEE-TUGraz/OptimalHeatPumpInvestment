import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import expost_model

def load_results_dataframe(scenario_index, model_formulation):
    """Load the results dataframe for a given scenario and model formulation."""
    file_path = os.path.join("results",f"scenario_{scenario_index}",f"{model_formulation}_results.xlsx")
    try:
        df_results = pd.read_excel(file_path, index_col=[0,1])
        return df_results
    except FileNotFoundError:
        print(f"Results file not found: {file_path}")
        return None

import json
import os

def load_key_results(scenario_index, model_formulation):
    """Load the key results dictionary for a given scenario and model formulation."""
    file_path = os.path.join(
        "results",
        f"scenario_{scenario_index}",
        f"{model_formulation}_key_results.json"
    )

    try:
        with open(file_path, "r") as f:
            key_results = json.load(f)
        return key_results

    except FileNotFoundError:
        print(f"Key results file not found: {file_path}")
        return None


def exact_hp_model(hp_capacity, p_el, PLR_min=0.1, COP=3.56):
    # Convert to numpy array for vectorized handling
    p_el = np.asarray(p_el)

    # Part Load Ratio (PLR)
    if hp_capacity == 0:
        plr = np.zeros_like(p_el)
    else:
        plr = p_el / hp_capacity

    # Part-load efficiency data (given)
    data_plr = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    eta_pl = np.array([1.0, 0.99775281, 0.99382022, 0.98820225, 0.98061798, 0.97078652, 0.95505618, 0.92247191, 0.86348315, 0.72331461])

    # Ensure increasing order for interpolation
    sort_idx = np.argsort(data_plr)
    data_PLR = data_plr[sort_idx]
    eta_PL = eta_pl[sort_idx]

    # Linear interpolation of efficiency
    eta_interp = np.interp(
        plr,
        data_PLR,
        eta_PL,
        left=0.0,  # below smallest PLR → no operation
        right=1.0  # above 1.0 → cap at nominal
    )

    # Enforce minimum PLR operation
    active = plr >= PLR_min

    # Heat output
    q_heat = np.zeros_like(plr, dtype=float)
    q_heat[active] = p_el[active] * COP * eta_interp[active]
    return q_heat


def calc_exact_heat_output(df_results: pd.DataFrame, dict_key_results: dict, set_hp = ["HP1","HP2"]):
    for hp in set_hp:
        hp_capacity = dict_key_results[f"HP_Investment_{hp}"]
        p_el = df_results[[f"Electricity_Consumption_kW_{hp}"]].values
        q_heat_exact = exact_hp_model(hp_capacity, p_el)
        df_results[[f"q_heat_exact_{hp}"]] = q_heat_exact

    # calculate total heat output
    q_heat_exact_total = np.zeros(df_results.shape[0])
    for hp in set_hp:
        q_heat_exact_total += df_results[[f"q_heat_exact_{hp}"]].values.flatten()
    df_results["q_heat_exact_total"] = q_heat_exact_total


def calc_actual_heat_deviation(df_results: pd.DataFrame, set_hp = ["hp1","hp2"]):
    heat_deviation = df_results["HeatDemand"] - df_results["q_heat_exact_total"] - df_results["Heat_Storage_Discharge_kW"] + df_results["Heat_Storage_Charge_kW"]

    # differentiate exess and non-supplied heat
    df_results["real_excess_heat"] = (-heat_deviation).clip(lower=0)
    df_results["real_non_supplied_heat"] = (heat_deviation).clip(lower=0)

    return None


def performe_complete_expost_analysis(scenario_index):
    model_formulations = ["LP", "UC", "PWL", "PWLR", "CR"]

    for model in model_formulations:
        df_results = load_results_dataframe(scenario_index, model)
        key_results = load_key_results(scenario_index, model)

        if df_results is not None and key_results is not None:
            calc_exact_heat_output(df_results, key_results, set_hp=["HP1","HP2"])
            calc_actual_heat_deviation(df_results, set_hp=["HP1","HP2"])

            save_expost_analysis_results(scenario_index, model, df_results)

def save_expost_analysis_results(scenario_index, model_formulation, df_results):
    """Save the updated results dataframe after ex-post analysis."""
    file_path = os.path.join("results",f"scenario_{scenario_index}",f"{model_formulation}_results_expost.xlsx")
    df_results.to_excel(file_path)
    print(f"Ex-post analysis results saved to: {file_path}")


def load_expost_analysis_results(scenario_index, model_formulation):
    """Load the ex-post analysis results dataframe for a given scenario and model formulation."""
    file_path = os.path.join("results",f"scenario_{scenario_index}",f"expost_{model_formulation}_results.xlsx")
    try:
        df_results = pd.read_excel(file_path, index_col=[0,1])
        return df_results
    except FileNotFoundError:
        print(f"Ex-post analysis results file not found: {file_path}")
        return None


def plot_performance_map(df_expost_results, dict_key_results, scenario_index, model_formulation):
    inv_hp1 = dict_key_results["HP_Investment_HP1"]  # kW_el
    inv_hp2 = dict_key_results["HP_Investment_HP2"]  # kW_el
    COP_ref = 3.56

    # ------------------------------------------------------------------
    # Figure setup (no shared axes!)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        ncols=2,
        figsize=(6.7, 3.0),
        sharex=False,
        sharey=False
    )

    ax_hp1, ax_hp2 = axes

    # ------------------------------------------------------------------
    # HP1 subplot
    # ------------------------------------------------------------------
    p_el_hp1 = df_expost_results["Electricity_Consumption_kW_HP1"]
    q_hp1 = df_expost_results["Heat_Produced_kW_HP1"]

    ax_hp1.scatter(
        p_el_hp1,
        q_hp1,
        alpha=0.8,
        s=3
    )

    # Reference COP line
    p_ref_hp1 = np.linspace(0, p_el_hp1.max() * 1.05, 100)
    ax_hp1.plot(
        p_ref_hp1,
        COP_ref * p_ref_hp1,
        linestyle="--",
        linewidth=1.2,
        color="black",
        label=r"$\mathrm{COP}=3.56$"
    )

    # Vertical line at investment capacity
    ax_hp1.axvline(
        inv_hp1,
        linestyle=":",
        linewidth=1.2,
        color="black",
        label=r"$P_\mathrm{el}^{\mathrm{inv}}$"
    )

    # Axis limits
    ax_hp1.set_xlim(0, max(p_el_hp1.max(), inv_hp1) * 1.1)
    ax_hp1.set_ylim(0, q_hp1.max() * 1.1)

    ax_hp1.set_title("HP1")
    ax_hp1.set_xlabel(r"$P_\mathrm{el}$ in kW")
    ax_hp1.set_ylabel(r"$Q_\mathrm{heat}$ in kW")
    ax_hp1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax_hp1.legend(frameon=False)

    # ------------------------------------------------------------------
    # HP2 subplot
    # ------------------------------------------------------------------
    p_el_hp2 = df_expost_results["Electricity_Consumption_kW_HP2"]
    q_hp2 = df_expost_results["Heat_Produced_kW_HP2"]

    ax_hp2.scatter(
        p_el_hp2,
        q_hp2,
        alpha=0.8,
        s=3
    )

    # Reference COP line
    p_ref_hp2 = np.linspace(0, p_el_hp2.max() * 1.05, 100)
    ax_hp2.plot(
        p_ref_hp2,
        COP_ref * p_ref_hp2,
        linestyle="--",
        linewidth=1.2,
        color="black",
        label=r"$\mathrm{COP}=3.56$"
    )

    # Vertical line at investment capacity
    ax_hp2.axvline(
        inv_hp2,
        linestyle=":",
        linewidth=1.2,
        color="black",
        label=r"$P_\mathrm{el}^{\mathrm{inv}}$"
    )

    # Axis limits
    ax_hp2.set_xlim(0, max(p_el_hp2.max(), inv_hp2) * 1.1)
    ax_hp2.set_ylim(0, q_hp2.max() * 1.1)

    ax_hp2.set_title("HP2")
    ax_hp2.set_xlabel(r"$P_\mathrm{el}$ in kW")
    ax_hp2.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax_hp2.legend(frameon=False)

    # ------------------------------------------------------------------
    # Layout & export
    # ------------------------------------------------------------------
    fig.suptitle(
        f"Heat Pump Performance Map ({model_formulation})",
        fontsize=10
    )

    fig.tight_layout(w_pad=1.2)
    fig.subplots_adjust(top=0.88)

    os.makedirs(
        os.path.join(
            "results",
            f"scenario_{scenario_index}",
            "plots"
        ),
        exist_ok=True
    )

    plot_path = os.path.join(
        "results",
        f"scenario_{scenario_index}",
        "plots",
        f"scenario_performance_map_{model_formulation}.pdf"
    )

    fig.savefig(plot_path, format="pdf", bbox_inches="tight")
    plt.close(fig)



def plot_all_performance_maps(scenario_index):
    model_formulations = ["LP", "UC", "PWL", "PWLR", "CR"]

    for model in model_formulations:
        df_results_expost = load_expost_analysis_results(scenario_index, model)
        dict_key_results = load_key_results(scenario_index, model)

        if df_results_expost is not None:
            plot_performance_map(df_results_expost, dict_key_results, scenario_index, model)




def plot_heat_balance_timeseries(
    df_expost_results,
    scenario_index,
    model_formulation,
    rp_to_plot,
    storage_capacity_kWh,
):
    # ------------------------------------------------------------------
    # Select representative period
    # ------------------------------------------------------------------
    df = df_expost_results.loc[rp_to_plot].copy()
    df = df.sort_index()

    n_steps = len(df)

    # ------------------------------------------------------------------
    # Time axis (quarter hours → hours)
    # ------------------------------------------------------------------
    dt = 0.25
    time_h = np.arange(n_steps) * dt
    bar_width = dt

    tick_positions = np.arange(0, time_h[-1] + 1e-9, 6.0)
    tick_labels = [f"{int(t):02d}:00" for t in tick_positions]

    # ------------------------------------------------------------------
    # Heat balance data
    # ------------------------------------------------------------------
    q_hp1 = df["Heat_Produced_kW_HP1"].values
    q_hp2 = df["Heat_Produced_kW_HP2"].values

    q_stor_dis = df["Heat_Storage_Discharge_kW"].values
    q_stor_ch  = df["Heat_Storage_Charge_kW"].values

    q_not_supplied = df["Heat_Not_Supplied_kW"].values
    q_excess       = df["Excess_Heat_Supplied_kW"].values

    q_demand = df["HeatDemand"].values

    # ------------------------------------------------------------------
    # Storage level (absolute)
    # ------------------------------------------------------------------
    stor_level_abs = df["Heat_Storage_Level_kWh"].values

    # ------------------------------------------------------------------
    # Figure & axes
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    x_bar = time_h

    bar_kws = dict(
        align="edge",
        edgecolor="black",
        linewidth=0.3,
    )

    # ------------------------------------------------------------------
    # Positive stacked bars (supply side)
    # ------------------------------------------------------------------
    bottom_pos = np.zeros(n_steps)

    ax.bar(x_bar, q_hp1, bar_width, bottom=bottom_pos,
           label=r"$q_{\mathrm{HP1}}$", **bar_kws)
    bottom_pos += q_hp1

    ax.bar(x_bar, q_hp2, bar_width, bottom=bottom_pos,
           label=r"$q_{\mathrm{HP2}}$", **bar_kws)
    bottom_pos += q_hp2

    ax.bar(x_bar, q_stor_dis, bar_width, bottom=bottom_pos,
           label=r"$q_{\mathrm{stor}}^{\mathrm{dis}}$", **bar_kws)
    bottom_pos += q_stor_dis

    ax.bar(x_bar, q_not_supplied, bar_width, bottom=bottom_pos,
           label=r"$q_{\mathrm{not\ supplied}}$", **bar_kws)

    # ------------------------------------------------------------------
    # Excess & storage charge ABOVE demand (hatched)
    # ------------------------------------------------------------------
    ax.bar(
        x_bar,
        q_stor_ch,
        bar_width,
        bottom=q_demand,
        label=r"$q_{\mathrm{stor}}^{\mathrm{ch}}$",
        hatch="//////",
        facecolor="none",
        **bar_kws,
    )

    ax.bar(
        x_bar,
        q_excess,
        bar_width,
        bottom=q_demand + q_stor_ch,
        label=r"$q_{\mathrm{excess}}$",
        hatch="\\\\\\\\\\\\",
        facecolor="none",
        **bar_kws,
    )

    # ------------------------------------------------------------------
    # Demand (stepwise)
    # ------------------------------------------------------------------
    ax.step(
        time_h,
        q_demand,
        where="post",
        color="black",
        linewidth=1.2,
        label=r"$q_{\mathrm{demand}}$",
        zorder=5,
    )

    # ------------------------------------------------------------------
    # Secondary axis: absolute storage level
    # ------------------------------------------------------------------
    ax2 = ax.twinx()
    ax2.plot(
        time_h,
        stor_level_abs,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label=r"$E_{\mathrm{stor}}$",
    )
    ax2.set_ylabel(r"Storage level in kWh")
    ax2.set_ylim(0, storage_capacity_kWh)

    # ------------------------------------------------------------------
    # Axes formatting
    # ------------------------------------------------------------------
    ax.set_xlabel("Time in h")
    ax.set_ylabel(r"$q$ in kW")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    y_supply_max = (q_hp1 + q_hp2 + q_stor_dis + q_not_supplied).max()
    y_over_max = (q_demand + q_stor_ch + q_excess).max()

    ax.set_xlim(time_h[0], time_h[-1] + dt)
    ax.set_ylim(0, 1.05 * max(y_supply_max, y_over_max))

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        frameon=False,
    )

    # ------------------------------------------------------------------
    # Title & export
    # ------------------------------------------------------------------
    ax.set_title(
        f"Heat balance – RP {rp_to_plot} ({model_formulation})",
        fontsize=10,
    )

    fig.tight_layout()

    plot_path_dir = os.path.join(
        "results",
        f"scenario_{scenario_index}",
        "plots",
    )
    os.makedirs(plot_path_dir, exist_ok=True)

    plot_path = os.path.join(
        plot_path_dir,
        f"heat_balance_timeseries_rp_{rp_to_plot}_{model_formulation}.pdf",
    )
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def plot_heat_balance_timeseries_all_rps(
            df_expost_results,
            scenario_index,
            model_formulation,
            storage_capacity_kWh,
    ):
        """
        Create heat balance plots for all representative periods (RPs)
        found in the index of df_expost_results.
        """

        # --------------------------------------------------------------
        # Extract unique representative periods
        # --------------------------------------------------------------
        if not isinstance(df_expost_results.index, pd.MultiIndex):
            raise ValueError(
                "df_expost_results must have a MultiIndex with rp as first level."
            )

        rps = df_expost_results.index.get_level_values(0).unique()

        print(f"Creating heat balance plots for {len(rps)} representative periods.")

        # --------------------------------------------------------------
        # Loop over RPs
        # --------------------------------------------------------------
        for rp in rps:
            try:
                plot_heat_balance_timeseries(
                    df_expost_results=df_expost_results,
                    scenario_index=scenario_index,
                    model_formulation=model_formulation,
                    rp_to_plot=rp,
                    storage_capacity_kWh=storage_capacity_kWh,
                )
            except Exception as e:
                print(f"⚠️ Failed to plot RP {rp}: {e}")

        print("All RP heat balance plots created.")

def plot_all_models(scenario_index, storage_capacity_kWh):
    model_formulations = ["LP", "UC", "PWL", "PWLR", "CR"]

    for model in model_formulations:
        df_results_expost = load_expost_analysis_results(scenario_index, model)

        if df_results_expost is not None:
            plot_heat_balance_timeseries_all_rps(
                df_expost_results=df_results_expost,
                scenario_index=scenario_index,
                model_formulation=model,
                storage_capacity_kWh=storage_capacity_kWh,
            )


########################################################################################################################
## Ex-Post analysis for costs
#########################################################################################################################

def load_electricity_price_df(scenario_index):
    """
    Returns DataFrame indexed by (rp, h) with column 'electricity_price'
    """
    file_path = os.path.join(
        "results",
        f"scenario_{scenario_index}",
        "electricity_price.xlsx",
    )
    return pd.read_excel(file_path, index_col=[0, 1])

def load_cop_scalor_df(scenario_index):
    """
    Returns DataFrame indexed by (rp, h) with column 'COP_scalor'
    """
    file_path = os.path.join(
        "results",
        f"scenario_{scenario_index}",
        "cop_scalor.xlsx",
    )
    return pd.read_excel(file_path, index_col=[0, 1])

def load_rp_weight_df(scenario_index):
    """
    Returns DataFrame with columns ['rp', 'weight']
    """
    file_path = os.path.join(
        "results",
        f"scenario_{scenario_index}",
        "rp_weights.xlsx",
    )
    return pd.read_excel(file_path)

def load_heat_demand(scenario_index):
    """
    Returns DataFrame indexed by (rp, h) with column 'HeatDemand'
    """
    file_path = os.path.join(
        "results",
        f"scenario_{scenario_index}",
        "heat_demand.xlsx",
    )
    df = pd.read_excel(file_path, index_col=[0, 1])
    return df

def calc_residual_demand(df: pd.DataFrame) -> None:
    df["residual_heat_demand"] = (
        df["HeatDemand"]
        - df["Heat_Storage_Discharge_kW"]
        + df["Heat_Storage_Charge_kW"]
    )

def invert_exact_hp_model(
    q_target,
    hp_capacity,
    PLR_min=0.1,
    COP=3.56,
    tol=1e-3,
    max_iter=50,
):
    if q_target <= 0 or hp_capacity == 0:
        return 0.0

    p_min = PLR_min * hp_capacity
    p_max = hp_capacity

    q_min = exact_hp_model(hp_capacity, p_min, PLR_min, COP)
    q_max = exact_hp_model(hp_capacity, p_max, PLR_min, COP)

    if q_target <= q_min:
        return p_min
    if q_target >= q_max:
        return p_max

    p_low, p_high = p_min, p_max
    for _ in range(max_iter):
        p_mid = 0.5 * (p_low + p_high)
        q_mid = exact_hp_model(hp_capacity, p_mid, PLR_min, COP)

        if abs(q_mid - q_target) < tol:
            return p_mid

        if q_mid < q_target:
            p_low = p_mid
        else:
            p_high = p_mid

    return p_mid


def reconstruct_hp_dispatch(
    df: pd.DataFrame,
    key_results: dict,
    set_hp=("HP1", "HP2"),
    PLR_min=0.1,
    COP=3.56,
):
    for hp in set_hp:
        df[f"p_el_actual_{hp}"] = 0.0
        df[f"q_heat_actual_{hp}"] = 0.0

    for (rp, h), row in df.iterrows():
        residual = row["residual_heat_demand"]

        hp_on = {
            hp: row[f"Electricity_Consumption_kW_{hp}"] > 0
            for hp in set_hp
        }
        active_hps = [hp for hp, on in hp_on.items() if on]

        if not active_hps:
            continue

        cap = {hp: key_results[f"HP_Investment_{hp}"] for hp in active_hps}
        p_el_orig = {
            hp: row[f"Electricity_Consumption_kW_{hp}"]
            for hp in active_hps
        }

        if len(active_hps) == 1:
            hp = active_hps[0]
            p_el = invert_exact_hp_model(
                residual, cap[hp], PLR_min, COP
            )
            df.at[(rp, h), f"p_el_actual_{hp}"] = p_el
            df.at[(rp, h), f"q_heat_actual_{hp}"] = exact_hp_model(
                cap[hp], p_el, PLR_min, COP
            )

        else:
            hp_adj = min(active_hps, key=lambda x: p_el_orig[x])
            hp_fix = max(active_hps, key=lambda x: p_el_orig[x])

            q_fix = exact_hp_model(
                cap[hp_fix],
                p_el_orig[hp_fix],
                PLR_min,
                COP,
            )

            residual_adj = max(residual - q_fix, 0)

            p_el_adj = invert_exact_hp_model(
                residual_adj, cap[hp_adj], PLR_min, COP
            )

            df.at[(rp, h), f"p_el_actual_{hp_fix}"] = p_el_orig[hp_fix]
            df.at[(rp, h), f"q_heat_actual_{hp_fix}"] = q_fix

            df.at[(rp, h), f"p_el_actual_{hp_adj}"] = p_el_adj
            df.at[(rp, h), f"q_heat_actual_{hp_adj}"] = exact_hp_model(
                cap[hp_adj], p_el_adj, PLR_min, COP
            )


def calc_expost_heat_mismatch(df: pd.DataFrame, set_hp=("HP1", "HP2")):
    q_total = sum(df[f"q_heat_actual_{hp}"] for hp in set_hp)
    heat_gap = df["residual_heat_demand"] - q_total

    df["real_excess_heat"] = (heat_gap).clip(lower=0)
    df["real_non_supplied_heat"] = (-heat_gap).clip(lower=0)


def calc_cyclic_startups(df: pd.DataFrame, set_hp=("HP1", "HP2")):
    for hp in set_hp:
        df[f"startup_{hp}"] = 0

        for rp in df.index.get_level_values(0).unique():
            idx = df.loc[rp].index
            p = df.loc[rp, f"p_el_actual_{hp}"].values

            startup = np.zeros(len(p), dtype=int)
            for t in range(len(p)):
                prev = p[t - 1] if t > 0 else p[-1]
                startup[t] = int(p[t] > 0 and prev == 0)

            df.loc[(rp, idx), f"startup_{hp}"] = startup

def calc_expost_costs(
    df: pd.DataFrame,
    df_price: pd.DataFrame,
    df_rp_weight: pd.DataFrame,
    key_results: dict,
    scenario_parameters: pd.DataFrame = None,
    set_hp=("HP1", "HP2"),
    DeltaH=0.25,
):
    # Map RP weights
    rp_weight = build_rp_weight_series(df, df_rp_weight)

    # Align electricity prices
    price = df_price["electricity_price"]

    # Electricity costs
    el_costs = 0.0
    for hp in set_hp:
        el_costs += (
            price
            * df[f"p_el_actual_{hp}"]
            * DeltaH
            * df.index.get_level_values(0).map(rp_weight)
        ).sum()


    # Startup costs
    startup_costs = 0.0
    for hp in set_hp:
        startup_costs += (
            scenario_parameters["StartUpCost"]
            * df[f"startup_{hp}"]
            * df.index.get_level_values(0).map(rp_weight)
        ).sum()

    # Penalties
    excess_costs = (
        scenario_parameters["EHSCost"]
        * df["real_excess_heat"]
        * DeltaH
        * df.index.get_level_values(0).map(rp_weight)
    ).sum()

    nsp_costs = (
        scenario_parameters["HNSCost"]
        * df["real_non_supplied_heat"]
        * DeltaH
        * df.index.get_level_values(0).map(rp_weight)
    ).sum()

    # Investment costs
    invest_costs = sum(
        key_results[f"HP_Investment_{hp}"] * scenario_parameters["InvestmentCost"] * scenario_parameters["DurationDays"] / 365
        for hp in set_hp
    )

    base_invest = sum(
        1 for hp in set_hp
        if key_results.get(f"HP_Investment_{hp}", 0) > 0
    )
    invest_costs += scenario_parameters["BaseInvestmentCost"] * base_invest * scenario_parameters["DurationDays"] / 365

    total_costs = (
        el_costs + startup_costs + excess_costs + nsp_costs + invest_costs
    )


    return {
        "total_costs": total_costs,
        "investment_costs": invest_costs,
        "electricity_costs": el_costs,
        "startup_costs": startup_costs,
        "excess_heat_costs": excess_costs,
        "non_supplied_heat_costs": nsp_costs,
        "total_excess_heat": (
            df["real_excess_heat"]
            * DeltaH
            * df.index.get_level_values(0).map(rp_weight)
        ).sum(),
        "total_non_supplied_heat": (
            df["real_non_supplied_heat"]
            * DeltaH
            * df.index.get_level_values(0).map(rp_weight)
        ).sum(),
    }


def build_rp_weight_series(df, df_rp_weight):
    rps = df.index.get_level_values(0).unique()

    if "rp" in df_rp_weight.columns:
        return df_rp_weight.set_index("rp")["weight"]

    if df_rp_weight.shape[0] == len(rps):
        return pd.Series(
            df_rp_weight["weight"].values,
            index=rps,
            name="weight",
        )

    raise ValueError("Cannot align RP weights with results dataframe")


def run_kpi_expost_analysis(scenario_index, scenario_parameter):
    model_formulations = ["LP", "UC", "PWL", "PWLR", "CR"]
    rows = []

    df_heatdemand = load_heat_demand(scenario_index)
    df_price = load_electricity_price_df(scenario_index)
    df_rp_weight = load_rp_weight_df(scenario_index)
    df_cop_scalor = load_cop_scalor_df(scenario_index)
    df_rp_weight.set_index("Unnamed: 0", inplace=True)

    for model in model_formulations:
        df = load_results_dataframe(scenario_index, model)
        key_results = load_key_results(scenario_index, model)

        if df is None or key_results is None:
            continue

        expost_results = expost_model.solve_expost_model(scenario_parameter, df_heatdemand, df_price, df_cop_scalor, df_rp_weight, key_results, model)

        # The KPI "SolveWork"/"Iterations" should report the INVESTMENT model's solve effort (what we
        # compare across formulations), not the ex-post validation solve. solve_expost_model returns the
        # ex-post solve work under these keys, so overwrite them with the investment model's values.
        expost_results["SolveWork"] = key_results.get("SolveWork")
        expost_results["Iterations"] = key_results.get("Iterations")

        rows.append({
            "scenario": scenario_index,
            "model": model,
            **expost_results,
        })

    return pd.DataFrame(rows)






if __name__ == "__main__":
    # Example usage
    scenario_index = "base_case_v4"  # specify the scenario index you want to analyze

    #performe_complete_expost_analysis(scenario_index)

    #plot_all_performance_maps(scenario_index)

    #plot_all_models(scenario_index, 50)


    #my_results = run_kpi_expost_analysis(scenario_index, scenar.loc[scenar["ScenarioIndex"]==scenario_index].squeeze())
    #my_results.to_excel(f"results/scenario_{scenario_index}/expost_kpi_results.xlsx", index=False)

    #print(my_results.head())

    # plot all performance maps and heat balance timeseries for all scenarios


    #df_temp = run_kpi_expost_analysis(scenario_index, scenario_parameter=scenar.loc[scenar["ScenarioIndex"]==scenario_index].squeeze())

    #print(df_temp.head())

    # plot all performance maps for the scenario
    plot_all_performance_maps(scenario_index)

    # plot all time plots
    plot_all_models(scenario_index, 22.4)