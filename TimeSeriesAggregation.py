import data
import pandas as pd
import numpy as np
import tsam.timeseriesaggregation as tsam



def performe_TSA(df_heat_demand, df_el_price, df_cop_scalor, noTypicalPeriods=4, hoursPerPeriod=24):
    ############### Time Series Aggregation ###############
    # merge both dataframes into one dataframe by joining on the index
    df_merged = df_heat_demand.join(
        [df_el_price, df_cop_scalor],
        how="inner"  # or "left", "right", "outer"
    )

    # indtroduve a datetime index starting from 2020-01-01 with 15-minute frequency
    date_range = pd.date_range(start="2018-01-01", periods=len(df_merged), freq="15min")
    df_merged.index = date_range


    aggregation = tsam.TimeSeriesAggregation(df_merged,
      noTypicalPeriods = noTypicalPeriods, # number of representative periods
      hoursPerPeriod = hoursPerPeriod, # durration of each period in hours (!)
      representationMethod = "distributionAndMinMaxRepresentation",
      distributionPeriodWise = False,
      clusterMethod = 'hierarchical'
    )

    # create the typical periods
    typPeriods = aggregation.createTypicalPeriods()
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


