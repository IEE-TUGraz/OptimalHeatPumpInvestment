import data
import pandas as pd
import numpy as np
import tsam.timeseriesaggregation as tsam


# Nominal heat-pump COP (actual COP = COP_NOMINAL * COP_scalor); matches the constant
# used in LP_model / UC_model. Actual electric power per step = Q_demand / (COP_NOMINAL * COP_scalor).
COP_NOMINAL = 3.56


########################################################################################################################
## Capacity-adequate design-period selection
##
## The distribution-based aggregation reproduces marginals/energy well but is not adequate for
## capacity sizing: it scrambles the per-step Q<->COP pairing and drops sustained extreme events.
## We therefore append one real "design period" -- the actual chronological block that needs the
## largest electric capacity -- so the LP is forced to size enough capacity to serve every real period.
##
## The selector uses a storage-AWARE capacity requirement (a small buffer shaves instantaneous spikes,
## so the binding block is the worst *sustained* event, which a raw-peak selector would miss). The
## buffer proxy mirrors the LP's storage dynamics (charge efficiency + self-discharge) but starts the
## buffer EMPTY -- a conservative boundary vs the LP's cyclic storage, so the selector cannot
## under-estimate the binding period (no full-resolution certification loop is run in this variant).
########################################################################################################################

def _feasible(C, Q, COP, storage_kwh, res_h, ch_eff, loss):
    """True if electric capacity C [kW] serves this block with zero unmet heat, buffer starting empty."""
    Q_hp = C * COP  # max heat deliverable per step [kW]
    soc = 0.0       # START-EMPTY (conservative vs the LP's cyclic buffer)
    for t in range(len(Q)):
        soc *= (1.0 - loss * res_h)  # self-discharge, matching the LP storage balance
        if Q[t] > Q_hp[t]:
            soc -= (Q[t] - Q_hp[t]) * res_h  # discharge to cover the shortfall (1:1)
            if soc < -1e-6:
                return False
        else:
            soc = min(storage_kwh, soc + ch_eff * (Q_hp[t] - Q[t]) * res_h)  # store surplus (excess dumped)
    return True


def _required_capacity(Q_period, S_period, cop_nominal, storage_kwh, res_h, ch_eff, loss):
    """Smallest electric capacity C [kW] that serves this block; monotonic in C -> binary search."""
    COP = cop_nominal * S_period
    lo, hi = 0.0, float((Q_period / COP).max())  # upper bound = peak elec with no storage (always feasible)
    for _ in range(60):
        C = 0.5 * (lo + hi)
        if _feasible(C, Q_period, COP, storage_kwh, res_h, ch_eff, loss):
            hi = C
        else:
            lo = C
    return hi


def required_capacity_per_period(df_merged, period_steps, cop_nominal, storage_kwh, res_h,
                                 ch_eff, loss, demand_col="Q_demand", cop_col="COP_scalor"):
    """Required electric capacity [kW] for every candidate (chronological) period."""
    n = len(df_merged) // period_steps
    Q = df_merged[demand_col].to_numpy()[:n * period_steps].reshape(n, period_steps)
    S = df_merged[cop_col].to_numpy()[:n * period_steps].reshape(n, period_steps)
    return np.array([
        _required_capacity(Q[k], S[k], cop_nominal, storage_kwh, res_h, ch_eff, loss)
        for k in range(n)
    ])


def select_design_period(df_merged, period_steps, cop_nominal, storage_kwh, res_h,
                         ch_eff, loss, demand_col="Q_demand", cop_col="COP_scalor"):
    """Return (index of the capacity-maximising candidate period, per-period required capacities)."""
    req_c = required_capacity_per_period(df_merged, period_steps, cop_nominal, storage_kwh,
                                         res_h, ch_eff, loss, demand_col, cop_col)
    d = int(np.argmax(req_c))
    return d, req_c


def count_infeasible_periods(df_merged, capacity_kw, period_steps, cop_nominal, storage_kwh, res_h,
                             ch_eff, loss, demand_col="Q_demand", cop_col="COP_scalor"):
    """Diagnostic (proxy, single ideal HP): how many candidate periods need more than `capacity_kw`."""
    req_c = required_capacity_per_period(df_merged, period_steps, cop_nominal, storage_kwh,
                                         res_h, ch_eff, loss, demand_col, cop_col)
    return int((req_c > capacity_kw + 1e-6).sum()), req_c


def _build_aggregation(df_merged, noTypicalPeriods, hoursPerPeriod):
    """
    Build the tsam aggregation with a real-profile (medoid) representation, so each typical
    period is an actual observed block with Q_demand and COP_scalor paired at every step.
    Prefers k_medoids (better-centred medoids); falls back to hierarchical clustering if the
    k_medoids MILP solver is unavailable. The pairing guarantee comes from the medoid
    representation, not from the clustering method. Returns (aggregation, typicalPeriods).
    """
    common = dict(
        noTypicalPeriods=noTypicalPeriods,
        hoursPerPeriod=hoursPerPeriod,
        representationMethod="medoidRepresentation",
        rescaleClusterPeriods=False,  # keep medoids as untouched real data -> exact Q<->COP pairing
    )
    try:
        aggregation = tsam.TimeSeriesAggregation(df_merged, clusterMethod="k_medoids", **common)
        typPeriods = aggregation.createTypicalPeriods()
        print("[TSA] clustering: k_medoids + medoidRepresentation (rescale off).")
        return aggregation, typPeriods
    except Exception as exc:
        print(f"[TSA] k_medoids unavailable ({type(exc).__name__}: {exc}); "
              f"falling back to hierarchical + medoidRepresentation.")
        aggregation = tsam.TimeSeriesAggregation(df_merged, clusterMethod="hierarchical", **common)
        typPeriods = aggregation.createTypicalPeriods()
        return aggregation, typPeriods


def _bulk_pairing_error(typPeriods_demand, typPeriods_cop_scalor, df_merged, period_steps,
                        bulk_labels, demand_col="Q_demand", cop_col="COP_scalor"):
    """
    Max mismatch between each bulk typical period and its nearest real (candidate) block, over
    Q_demand and COP_scalor. ~0 confirms every typical period is an actual observed, correctly-paired
    block (so P_el = Q_demand/(cop_nominal*COP_scalor) matches a real operating point at every step).
    """
    n = len(df_merged) // period_steps
    Qc = df_merged[demand_col].to_numpy()[:n * period_steps].reshape(n, period_steps)
    Sc = df_merged[cop_col].to_numpy()[:n * period_steps].reshape(n, period_steps)
    worst = 0.0
    for rp in bulk_labels:
        q = typPeriods_demand.loc[rp, demand_col].to_numpy()
        s = typPeriods_cop_scalor.loc[rp, cop_col].to_numpy()
        # nearest real block (a medoid equals one of the candidate blocks exactly)
        dist = np.abs(Qc - q).max(axis=1) + np.abs(Sc - s).max(axis=1)
        k = int(np.argmin(dist))
        worst = max(worst, float(np.abs(Qc[k] - q).max()), float(np.abs(Sc[k] - s).max()))
    return worst


def performe_TSA(df_heat_demand, df_el_price, df_cop_scalor, noTypicalPeriods=4, hoursPerPeriod=24,
                 storage_capacity_kwh=0.0, storage_ch_eff=1.0, storage_loss=0.0,
                 resolution_h=0.25, cop_nominal=COP_NOMINAL, add_design_period=True,
                 design_label="rp_design"):
    ############### Time Series Aggregation ###############
    # merge both dataframes into one dataframe by joining on the index
    df_merged = df_heat_demand.join(
        [df_el_price, df_cop_scalor],
        how="inner"  # or "left", "right", "outer"
    )

    # indtroduve a datetime index starting from 2020-01-01 with 15-minute frequency
    date_range = pd.date_range(start="2018-01-01", periods=len(df_merged), freq="15min")
    df_merged.index = date_range


    # Real-profile (medoid) representation: every typical period is an actual observed 48-h block,
    # so Q_demand and COP_scalor stay paired at each step (unlike the distribution representation,
    # which sorts each column independently and breaks the pairing). rescaleClusterPeriods=False keeps
    # the medoid as untouched real data, so Q_demand / (cop_nominal * COP_scalor) stays exactly
    # consistent. Falls back to hierarchical clustering if k_medoids' MILP solver is unavailable.
    aggregation, typPeriods = _build_aggregation(df_merged, noTypicalPeriods, hoursPerPeriod)
    # capture the cluster assignment of every candidate period BEFORE it goes out of scope
    cluster_order = np.asarray(aggregation.clusterOrder)
    # rename the two index elements to rp and h
    typPeriods.index.names = ['rp', 'time_h']

    # add for the rp index for each elemet a prefix rp to the numbers
    typPeriods = typPeriods.reset_index()
    typPeriods['rp'] = typPeriods['rp'].apply(lambda x: f"rp{str(x + 1).zfill(2)}")
    typPeriods['time_h'] = typPeriods['time_h'].apply(lambda x: f"h{str(x+1).zfill(6)}")
    typPeriods = typPeriods.set_index(['rp', 'time_h'])

    # create a copy for the dataframe demand only
    typPeriods_demand = typPeriods[['Q_demand']].copy()
    typPeriods_electricityPrice = typPeriods[['electricity_price']].copy()
    typPeriods_cop_scalor = typPeriods[['COP_scalor']].copy()


    # get the weights as a pandas Dataframe
    weights = aggregation.clusterPeriodNoOccur
    df_rpWeights = pd.DataFrame.from_dict(weights, orient='index', columns=['weight'])

    # add the prefix rp to the index
    df_rpWeights.index = df_rpWeights.index.map(lambda x: f"rp{str(x + 1).zfill(2)}")

    # Pairing check: every bulk typical period must be an actual observed block with Q<->COP paired.
    period_steps = int(round(hoursPerPeriod / resolution_h))
    pairing_err = _bulk_pairing_error(typPeriods_demand, typPeriods_cop_scalor, df_merged,
                                      period_steps, list(df_rpWeights.index))
    print(f"[TSA] bulk-period pairing error (max real-block mismatch in Q & COP): {pairing_err:.3e}")
    assert pairing_err < 1e-6, (
        f"typical periods are not correctly-paired real blocks (pairing error {pairing_err:.3e}); "
        f"medoid representation expected")

    if add_design_period:
        (typPeriods_demand, typPeriods_electricityPrice, typPeriods_cop_scalor,
         df_rpWeights) = _append_design_period(
            df_merged, cluster_order, typPeriods_demand, typPeriods_electricityPrice,
            typPeriods_cop_scalor, df_rpWeights, noTypicalPeriods, hoursPerPeriod,
            storage_capacity_kwh, storage_ch_eff, storage_loss, resolution_h,
            cop_nominal, design_label,
        )

    return typPeriods_demand, typPeriods_electricityPrice, typPeriods_cop_scalor, df_rpWeights


def _append_design_period(df_merged, cluster_order, typPeriods_demand, typPeriods_electricityPrice,
                          typPeriods_cop_scalor, df_rpWeights, noTypicalPeriods, hoursPerPeriod,
                          storage_capacity_kwh, storage_ch_eff, storage_loss, resolution_h,
                          cop_nominal, design_label):
    """
    Append the real capacity-maximising 48-h block as an extra 'design' representative period and
    reweight occurrences so annual energy is approximately preserved. Keeps the clustered (medoid)
    periods untouched; only adds one more pristine, correctly-paired period.
    """
    period_steps = int(round(hoursPerPeriod / resolution_h))
    n_candidates = len(df_merged) // period_steps

    d, req_c = select_design_period(df_merged, period_steps, cop_nominal, storage_capacity_kwh,
                                    resolution_h, storage_ch_eff, storage_loss)
    c = int(cluster_order[d])                 # cluster the design period belongs to (0-based)
    c_label = f"rp{c + 1:02d}"                # its rp label in the existing convention

    print(f"[TSA] design period: candidate {d} (of {n_candidates}), required capacity "
          f"{req_c[d]:.1f} kW; belongs to cluster {c_label}. "
          f"Max clustered-period required capacity: {req_c.max():.1f} kW.")

    # --- raw, chronological, correctly-paired design block ---
    block = df_merged.iloc[d * period_steps:(d + 1) * period_steps]
    time_labels = [f"h{str(i + 1).zfill(6)}" for i in range(period_steps)]
    idx = pd.MultiIndex.from_arrays([[design_label] * period_steps, time_labels],
                                    names=['rp', 'time_h'])
    design_demand = pd.DataFrame({'Q_demand': block['Q_demand'].to_numpy()}, index=idx)
    design_price = pd.DataFrame({'electricity_price': block['electricity_price'].to_numpy()}, index=idx)
    design_cop = pd.DataFrame({'COP_scalor': block['COP_scalor'].to_numpy()}, index=idx)

    # --- reweight: move one occurrence from the design period's cluster to the design period ---
    df_rpWeights.loc[c_label, 'weight'] -= 1

    # edge case: cluster emptied -> its only member was the design period; drop it to avoid double count
    if df_rpWeights.loc[c_label, 'weight'] == 0:
        df_rpWeights = df_rpWeights.drop(index=c_label)
        typPeriods_demand = typPeriods_demand.drop(index=c_label, level='rp')
        typPeriods_electricityPrice = typPeriods_electricityPrice.drop(index=c_label, level='rp')
        typPeriods_cop_scalor = typPeriods_cop_scalor.drop(index=c_label, level='rp')
        print(f"[TSA] cluster {c_label} emptied by the design-period extraction; dropped it.")

    # NOTE: the decremented cluster's medoid still counts the design member in its occurrences
    # (a ~1/N second-order effect); left as-is -> slightly conservative and simpler (see brief).

    typPeriods_demand = pd.concat([typPeriods_demand, design_demand])
    typPeriods_electricityPrice = pd.concat([typPeriods_electricityPrice, design_price])
    typPeriods_cop_scalor = pd.concat([typPeriods_cop_scalor, design_cop])
    df_rpWeights.loc[design_label, 'weight'] = 1

    # --- invariants ---
    assert (df_rpWeights['weight'] >= 0).all(), "negative representative-period weight"
    assert int(df_rpWeights['weight'].sum()) == n_candidates, (
        f"weights sum {df_rpWeights['weight'].sum()} != candidate periods {n_candidates}")
    assert design_label in df_rpWeights.index and df_rpWeights.loc[design_label, 'weight'] == 1

    # represented annual heat: medoid + rescaleClusterPeriods=False reproduces energy only
    # approximately (no per-column mean forcing), so this is a tolerance check that warns, not fails.
    energy_tol = 0.03  # ~3%
    raw_heat = df_merged['Q_demand'].sum() * resolution_h
    rep_heat = sum(
        df_rpWeights.loc[rp, 'weight'] * typPeriods_demand.loc[rp, 'Q_demand'].sum() * resolution_h
        for rp in df_rpWeights.index
    )
    dev = abs(rep_heat - raw_heat) / raw_heat
    info = (f"represented annual heat deviation from raw: {dev*100:.2f}% "
            f"({len(df_rpWeights)} representative periods, weights sum {int(df_rpWeights['weight'].sum())}).")
    if dev > energy_tol:
        print(f"[TSA] WARNING: {info} Exceeds {energy_tol*100:.0f}% tolerance "
              f"(expected with medoid representation; not fatal).")
    else:
        print(f"[TSA] {info}")

    return typPeriods_demand, typPeriods_electricityPrice, typPeriods_cop_scalor, df_rpWeights


def adjust_format_chrono(df_heat_demand, df_el_price, df_cop_scalor):
    # create a new index with only one representative period
    df_heat_demand_chrono = df_heat_demand.copy()
    df_el_price_chrono = df_el_price.copy()
    df_cop_scalor_chrono = df_cop_scalor.copy()

    # reset the index to have time_h as a column
    df_heat_demand_chrono = df_heat_demand_chrono.reset_index()
    df_el_price_chrono = df_el_price_chrono.reset_index()
    df_cop_scalor_chrono = df_cop_scalor_chrono.reset_index()

    # create a new column rp with only one value rp01
    df_heat_demand_chrono['rp'] = 'rp01'
    df_el_price_chrono['rp'] = 'rp01'
    df_cop_scalor_chrono['rp'] = 'rp01'

    # set the index to rp and time_h
    df_heat_demand_chrono = df_heat_demand_chrono.set_index(['rp', 'time_h'])
    df_el_price_chrono = df_el_price_chrono.set_index(['rp', 'time_h'])
    df_cop_scalor_chrono = df_cop_scalor_chrono.set_index(['rp', 'time_h'])

    # create weights dataframe with only one representative period with weight 1
    df_rpWeights_chrono = pd.DataFrame({'weight': [1]}, index=['rp01'])

    return df_heat_demand_chrono, df_el_price_chrono, df_cop_scalor_chrono, df_rpWeights_chrono


