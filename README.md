# Optimal Heat Pump Investment Considering Part-Load Efficiency

## Overview

This repository contains the code and data accompanying the research on
optimal investment planning of heat pump systems while explicitly
accounting for part-load efficiency effects.

Heat pumps exhibit efficiency variations depending on their operating
point. In particular, part-load behaviour can significantly influence
both optimal system sizing and operational performance. Conventional
linear optimisation models often neglect these effects, which may lead
to structurally ambiguous investment decisions or suboptimal system
configurations.

The framework provided in this repository enables the joint optimisation
of investment and operation of multiple heat pumps under different
mathematical formulations representing part-load behaviour. The
implementation supports:

-   comparison of five alternative optimisation formulations
-   integrated capacity expansion and operational optimisation
-   time series aggregation using representative periods
-   ex-post validation of investment decisions on full-resolution time
    series
-   flexible adaptation to different datasets and system configurations

The code is intended both for research purposes and as a starting point
for practical system design studies.

------------------------------------------------------------------------

## Repository Structure and Capabilities

The workflow automatically:

1.  performs time series aggregation based on representative periods
2.  solves all defined scenarios using five different model formulations
3.  performs ex-post operational validation for fixed investment
    capacities
4.  generates plots and evaluation metrics
5.  stores all intermediate and final results for each scenario

The framework is designed to be adaptable to different heat demand
profiles, electricity prices, and system configurations.

------------------------------------------------------------------------

## Usage

### 1. Clone or download Repository

```bash
git clone https://github.com/IEE-TUGraz/OptimalHeatPumpInvestment.git
cd OptimalHeatPumpInvestment
```

### 2. Create and Activate Environment

```bash
conda env create -f environment.yml conda activate heatpump_env
```

### 3. Configure Parameters

parameter.yaml

Example parameters include:

-   number of representative periods


### 4. Define Scenarios

inputScenarios.xlsx

Each scenario will automatically be solved using all five model
formulations.

### 5. Run Optimisation

python main.py

------------------------------------------------------------------------

## Outputs

For each scenario, the framework generates:

-   optimal investment decisions
-   operational schedules
-   ex-post validation results
-   comparison of model formulations
-   plots and evaluation metrics

All results are stored in the:

results/

directory.

------------------------------------------------------------------------

## Hints and Customisation

Solver

The default solver is Gurobi, which requires a valid license. The solver
can be changed in the code if alternative solvers are preferred.

Using custom data

To use different input data, provide a dataframe with the required
column structure as defined in the code and replace the data loading
section in main.py.

Reducing computation time

If only a single formulation is of interest (e.g. LP or PWLR), the
other code blocks can be commented out in main.py. This can
significantly reduce computation time.

Improving solver performance

If good estimates for investment sizes are available, bounds can be
tightened in:

initialise_investment_bounds()

This may considerably speed up optimisation.

------------------------------------------------------------------------

## License

This project is distributed under the MIT License.
