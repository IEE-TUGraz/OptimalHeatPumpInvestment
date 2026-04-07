import pandas as pd
import yaml
import os

def load_input_scenarios(file_path):
    """Load input scenarios from a Excel file."""
    input_scenarios = pd.read_excel(file_path, skiprows=[1])
    print(f"Loaded {len(input_scenarios)} scenarios successfully.")

    return input_scenarios

def load_data(file_path):
    data = pd.read_pickle(file_path)

    print(f"Data loaded successfully from {file_path}")

    return data


def extract_relevant_data(scenario_data, parameter):
    start_index = parameter["StartDay"] * 24 * 4  # 15 min resolution
    duration_hours = parameter["DurationDays"] * 24
    end_index = start_index + duration_hours * 4  # 15 min resolution

    # extract just casestudy timespan
    data_cs = scenario_data.iloc[start_index:end_index, :].copy()

    # add the time index column with h0001, h0002, ...
    data_cs.insert(0, "time_h", [f"h{str(i + 1).zfill(6)}" for i in range(len(data_cs))])

    # extract relecant dataframes
    df_heat_demand = data_cs.loc[:, ["time_h", "Q_heating"]].set_index("time_h").rename(columns={"Q_heating": "Q_demand"})
    if str(parameter["VariableElCost"]).strip().lower() == "true":
        df_el_price = data_cs.loc[:, ["time_h", "Electricity_Price"]].set_index("time_h").rename(columns={"Electricity_Price": "electricity_price"})
        print("Using variable electricity cost for the scenario.")
    elif str(parameter["VariableElCost"]).strip().lower() == "false":
        df_el_price = data_cs.loc[:, ["time_h", "Electricity_Price_Const"]].set_index("time_h").rename(columns={"Electricity_Price_Const": "electricity_price"})
        print("Using constant electricity cost for the scenario.")

    # rescale the eletrichty price by a factor if specified
    df_el_price["electricity_price"] = df_el_price["electricity_price"] * parameter["ElPriceScalor"]

    # extract the COP scaling factors
    df_cop_scalor = data_cs.loc[:, ["time_h", "COP_scalor"]].set_index("time_h")

    return df_heat_demand, df_el_price, df_cop_scalor


def save_data_to_disk(df_el_price, df_rpWeights, df_heat_demand, df_cop_scalor, scenario):
    path = os.path.join("results", "scenario_" + str(scenario['ScenarioIndex']))

    if not os.path.exists(path):
        os.makedirs(path)


    df_el_price.to_excel(os.path.join(path,"electricity_price.xlsx"))
    df_rpWeights.to_excel(os.path.join(path,"rp_weights.xlsx"))
    df_heat_demand.to_excel(os.path.join(path,"heat_demand.xlsx"))
    df_cop_scalor.to_excel(os.path.join(path,"cop_scalor.xlsx"))

    return None

def load_parameter():
    # load the parameter from the yaml file
    with open("parameter.yaml", "r") as file:
        parameter = yaml.safe_load(file)
    print("Parameters loaded successfully from parameters.yaml")
    return parameter

def save_rp_weights(df_rpWeights, parameter):
    # extract the representative period weights and save them to a csv file
    if not os.path.exists("results"):
        os.makedirs("results")

    scenario_folder = os.path.join("results", "scenario_" + str(parameter['ScenarioIndex']))
    if not os.path.exists(scenario_folder):
        os.makedirs(scenario_folder)

    path = os.path.join(scenario_folder, "rp_weights.csv")    

    df_rpWeights.to_csv(path, index=True)
    print("Representative period weights saved to rp_weights.csv")