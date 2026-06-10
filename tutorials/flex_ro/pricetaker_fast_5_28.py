##########
#
# Pricetaker Implementation for WRD plant using surrogate models for UF and RO energy consumption.
# Short = Only testing one or two days, and no rain events or demand response events.
#
##########

import warnings
import logging
import os
from datetime import datetime

warnings.filterwarnings("ignore", message=".*implicit domain of 'Any'.*")
logging.getLogger("pyomo").setLevel(logging.ERROR)
from idaes.apps.grid_integration import PriceTakerModel
import pandas as pd
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.environ import SolverFactory
from pathlib import Path

from watertap.flowsheets.flex_desal import wrd_ro_flowsheet as fs
from watertap.flowsheets.flex_desal import utils
from watertap.flowsheets.flex_desal.params import FlexDesalParams
from watertap.core.solvers import get_solver

from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.core.util.model_statistics import degrees_of_freedom


def main(season, flex_type, num_flexible_trains=4):
    season_map = {
        "summer": "price_signals/wrd_pricesignal_summer_week.csv",
        "winter": "price_signals/wrd_pricesignal_winter_week.csv",
    }
    season_key = season.lower()
    if season_key not in season_map:
        raise ValueError(
            f"Invalid season '{season}'. Valid options are: {sorted(season_map)}"
        )

    flex_type_key = flex_type.lower()
    valid_flex_types = {"rr", "flow", "both", "no_flex"}
    if flex_type_key not in valid_flex_types:
        raise ValueError(
            "Invalid flex_type "
            f"'{flex_type}'. Valid options are: {sorted(valid_flex_types)}"
        )

    selected_price_signal_stem = Path(season_map[season_key]).stem
    output_suffix = (
        f"{season_key}_{flex_type_key}_{num_flexible_trains}_flexible_trains"
    )
    if selected_price_signal_stem.upper().endswith("RTP"):
        output_suffix = f"{output_suffix}_RTP"
    if selected_price_signal_stem.upper().endswith("TOU_8"):
        output_suffix = f"{output_suffix}_TOU_8"
    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    # Load price data
    price_data = pd.read_csv(script_dir / season_map[season_key])
    price_data["Energy Rate"] = (
        price_data["electric_energy_on_peak"]
        + price_data["electric_energy_mid_peak"]
        + price_data["electric_energy_off_peak"]
        + price_data["electric_energy_super_off_peak"]
    )
    price_data["Fixed Demand Rate"] = price_data["electric_demand_fixed"]
    price_data["Var Demand Rate"] = price_data["electric_demand_peak"]
    price_data["Customer Cost"] = price_data["electric_customer_fixed_charge"]
    price_data["Demand_Response_Price"] = price_data["electric_demand_response_price"]

    price_data["Emissions Intensity"] = 0
    peak_hours = price_data["Var Demand Rate"].to_numpy() != 0

    # Load PV data
    pv_kW = price_data["solar_output_kW"]
    pv_capacity = max(pv_kW)
    pv_capacity_factors = pv_kW / pv_capacity

    m = PriceTakerModel()
    # Find start and end datetimes and time step  from the price data
    price_datetimes = pd.to_datetime(price_data["DateTime"])
    data_start = price_datetimes.iloc[0]
    data_next_time = price_datetimes.iloc[1]
    timestep_hours = (data_next_time - data_start).total_seconds() / 3600
    start_date = data_start.strftime("%Y-%m-%d %H:%M:%S")
    end_date = price_datetimes.iloc[-1].strftime("%Y-%m-%d %H:%M:%S")

    # Instantiate an object containing the model parameters
    m.params = FlexDesalParams(
        start_date=start_date,
        end_date=end_date,
        annual_production_AF=12000,
        timestep_hours=timestep_hours,
        include_onsite_solar=False,
        onsite_capacity=pv_capacity,
        nonworking_hours=list(range(0, 8))
        + list(
            range(18, 24)
        ),  # 6pm-8am are nonworking hours (assuming time index starts at 0 for 12am-1am)
        # rainy_days=1,  # This will reduce the maxumim value for annual_production AF
        CAPEX_yr=6498300,  # For WRD, this assumes a 30 yr lifetime
        include_demand_response=True,
        max_daily_shutdowns=1,  # I'd like to change to one a day
    )

    m.params.intake.update(
        {
            "energy_intensity": 0,
            "nominal_flowrate": 2500,
            "feed_cost": 0.16,
            "chemical_cost": 0.0332,
        }
    )  # m3/hr

    m.params.wrd_uf.update(
        {
            "minimum_downtime": 2,
            "startup_delay": 2,
            "minimum_flowrate": 344,  # m3/hr
            "nominal_flowrate": 900,
            "maximum_flowrate": 989,
            "surrogate_type": "quadratic_energy_intensity",
            "surrogate_a": 2.71e-1,
            "surrogate_b": -3.32e-4,
            "surrogate_c": 2.39e-7,
            "nominal_recovery": 0.96,
            "num_uf_pumps": 3,
        }
    )

    m.params.wrd_ro.update(
        {
            "startup_delay": 2,  # hours
            "minimum_downtime": 2,  # hours
            "minimum_flowrate": 520,  # m3/hr
            "nominal_flowrate": 602,
            "maximum_flowrate": 635,
            "allow_variable_recovery": flex_type_key not in {"flow", "no_flex"},
            "surrogate_type": "PySMO_polyfit",
            "surrogate_file": script_dir / "ro_SEC_poly_fit_order_1.json",
            "minimum_recovery": 0.88,
            "nominal_recovery": 0.925,
            "maximum_recovery": 0.925,
            "num_ro_skids": 4,
            "replacement_types": ["membranes", "motors"],
            "replacement_costs": [
                500 * 4 * (72 + 30 + 15),
                125000,
            ],  # $ per replacement
            "replacement_lifetimes": [5, 20],  # years
            "replacement_max_flex_penalty": [
                0.1,
                0.1,
            ],  # Reduction in lifetime if shutdowns occur every day (?)
        }
    )

    m.params.posttreatment.update(
        {
            "energy_intensity": 0.101,
            "chemical_cost": 0.0310,
        }
    )  # kWh/m3 #$/m3

    m.params.brinedischarge.update({"brine_cost": 0.43, "energy_intensity": 0})

    # Append LMP data to the model
    m.append_lmp_data(lmp_data=price_data["Energy Rate"])

    m.build_multiperiod_model(
        flowsheet_func=fs.build_desal_flowsheet,
        flowsheet_options={"params": m.params},
    )

    fs.restrict_flexible_trains(m, num_flexible_trains=num_flexible_trains)

    fs.begin_and_end_constraint(m)

    if season_key == "summer":
        fs.fix_operations_for_first_four_days(m, peak_hours=peak_hours)

    # Update the time-varying parameters other than the LMP, such as
    # demand costs and emissions intensity. LMP value is updated by default
    m.update_operation_params(
        {
            "fixed_demand_rate": price_data["Fixed Demand Rate"],
            "variable_demand_rate": price_data["Var Demand Rate"],
            "emissions_intensity": price_data["Emissions Intensity"],
            "customer_cost": price_data["Customer Cost"],
            "demand_response_price": price_data["Demand_Response_Price"],
        }
    )
    if m.params.include_onsite_solar:
        m.update_operation_params(
            {"power_generation.capacity_factor": pv_capacity_factors}
        )

    # Add demand cost and fixed cost calculation constraints
    fs.add_demand_and_fixed_costs(m)

    # Add the startup delay constraints
    fs.add_delayed_startup_constraints(m)
    # fs.add_delayed_shutdown_constraints(m)
    # fs.repeat_weekdays(m)

    m.total_water_production = pyo.Expression(
        expr=m.params.timestep_hours
        * sum(m.period[:, :].posttreatment.product_flowrate)
    )
    m.total_energy_cost = pyo.Expression(expr=sum(m.period[:, :].energy_cost))

    # Demand costs are automatically normalized by number of months. So for a sample week, it multiplies by 7/31.
    m.total_demand_cost = pyo.Expression(
        expr=m.fixed_demand_cost + m.variable_demand_cost
    )
    m.total_customer_cost = pyo.Expression(
        expr=sum(m.period[:, :].customer_cost) * m.params.num_months
    )

    fs.add_flow_costs(m)  # Flow costs = Feed, Brine, and Chemicals
    fs.add_useful_expressions(m)
    # This adds the total_demand_response_revenue, which only represents one of the available SCE DR options.

    m.total_op_cost = pyo.Expression(
        expr=m.total_energy_cost
        + m.total_demand_cost
        + m.total_customer_cost
        - m.total_demand_response_revenue
        + m.total_feed_cost
        + m.total_brine_cost
        + m.total_chemical_cost
    )
    # add CAPEX as a fixed cost to calculate LCOW
    m.fixed_cost = pyo.Expression(expr=m.params.CAPEX_yr * m.params.num_months / 12)
    m.total_cost = pyo.Expression(expr=m.total_op_cost + m.fixed_cost)

    m.LCOW = pyo.Expression(expr=m.total_cost / m.total_water_production)  # $/m3

    fs.constrain_water_production(m)

    # If water recovery is static, it must be fixed
    if not m.params.wrd_ro.allow_variable_recovery:
        utils.wrd_fix_ro_recovery(
            m,
            ro_recovery=m.params.wrd_ro.nominal_recovery,
        )
    # Always want to fix the UF recovery
    utils.wrd_fix_uf_recovery(
        m,
        uf_recovery=m.params.wrd_uf.nominal_recovery,
    )

    if flex_type_key == "rr":
        _fix_nominal_flowrates(m)

    # Could cause feasibility issues b/c this is a slack variable essentially.
    # m.fix_operation_var("reverse_osmosis.leftover_flow", 0)

    # Flowrates not fixed, but shouldn't randomly fluctuate either.
    fs.add_flow_changes_penalty_binary(m)

    # restricts number of shutdowns per 24 hours period, mainly to reduce solution space
    # fs.add_maximum_shutdowns(m)

    # fs.add_working_hours_constraint(m)

    # fs.add_rain_shutdowns(m)

    # This does not include the replacement costs atm because they don't drive the optimization. Also I removed the flexibility penalty
    m.obj = pyo.Objective(
        expr=1e-4
        * (
            m.total_energy_cost
            + m.total_demand_cost
            + m.total_customer_cost
            - m.total_demand_response_revenue
            + m.total_feed_cost
            + m.total_brine_cost
            + m.total_chemical_cost
            + m.flow_changes_penalty
        ),
        sense=pyo.minimize,
    )

    # Only to find the baseline power for this water production
    if flex_type_key == "no_flex" and num_flexible_trains == 0:
        m.enforce_steady_state = pyo.Constraint(expr=m.flow_changes_penalty == 0)

    print(degrees_of_freedom(m))

    # dt = DiagnosticsToolbox(m)
    # dt.report_structural_issues()

    # IPOPT
    # solver = get_solver()

    mip_gap = 0.01
    solver = pyo.SolverFactory("gurobi_direct_minlp")
    solver.options["MIPGap"] = mip_gap  # 1.0 %
    # solver.options["MIPGapAbs"] = (
    #     0.1  # $1,000 (b/c objective function is scaled down by 1e-4)
    # )
    # solver.options["MIPFocus"] = 1
    results = solver.solve(m, tee=True)

    print(f"m.flow_changes_penalty(): {m.flow_changes_penalty()}")
    print(f"Total operational cost: {m.total_op_cost():.2f}")

    pyo.assert_optimal_termination(results)

    # Baseline power is a function of the target water production, but needs to be calculated by running this model!
    if season_key == "winter":
        baseline_electricity_cost = 25005  # $
    else:
        baseline_electricity_cost = 50843  # $/kWh

    fs.calculate_replacement_costs(m)
    fs.calculate_flexibility_metrics(
        m,
        baseline_power=1127,
        baseline_electricity_cost=baseline_electricity_cost,
        baseline_replacement_cost=992,
    )  # 1080 is for 1200 AF yearly target

    design_var_values = m.get_design_var_values()
    filtered_design_var_values = {
        k: v
        for k, v in design_var_values.items()
        if "flow_change" not in k and "flow_changed" not in k and "reduction" not in k
    }
    print(filtered_design_var_values)

    # Write optimal values of all operational variables to a csv file
    output_csv = script_dir / f"wrd_result_{output_suffix}.csv"
    m.get_operation_var_values().to_csv(output_csv)
    print(f"Saved operation variable results to: {output_csv}")

    plot_function(
        m,
        n_time_points=len(price_data),
        output_stem=script_dir / f"wrd_pricetaker_{output_suffix}",
        peak_hours=peak_hours,
    )

    # # Plot operational variables
    # fig, axs = m.plot_operation_profile(
    #     operation_vars=[
    #         "fixed_demand_rate",
    #         "variable_demand_rate",
    #         "posttreatment.product_flowrate",
    #         "num_skids_online",
    #     ],
    # )
    # fig.savefig(script_dir / f"wrd_operation_profile_{output_suffix}.png")

    return m


if __name__ == "__main__":
    seasons = ["summer"]
    flex_types = ["both"]
    num_flex_skids = [4]

    results_rows = []

    for season in seasons:
        for flex_type in flex_types:
            for num_skids in num_flex_skids:
                m = main(
                    season=season, flex_type=flex_type, num_flexible_trains=num_skids
                )
                results_rows.append(
                    {
                        "Season": season,
                        "Flexibility Type": flex_type,
                        "Num Flexible Trains": num_skids,
                        "Total Operational Cost": m.total_op_cost(),
                        "Total Water Production (m3)": m.total_water_production(),
                        "LCOW ($/m3)": m.LCOW(),
                        "Total Energy Cost": m.total_energy_cost(),
                        "Fixed Demand Cost": m.fixed_demand_cost(),
                        "Variable Demand Cost": m.variable_demand_cost(),
                        "Total Electricity Cost": m.total_energy_cost()
                        + m.total_demand_cost(),
                        "Total Feed Cost": m.total_feed_cost(),
                        "Total Brine Cost": m.total_brine_cost(),
                        "Total Chemical Cost": m.total_chemical_cost(),
                        "Total Replacement Cost": m.total_replacement_cost(),
                        "Total Demand Response Revenue": m.total_demand_response_revenue(),
                        "Total Cost": m.total_cost(),
                        "Maximum Power": m.maximum_power(),
                        "Energy Capacity": m.energy_capacity(),
                        "Power Capacity": m.power_capacity(),
                        "LVOF": m.LVOF(),
                    }
                )

    results_df = pd.DataFrame(results_rows)
    script_dir = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = script_dir / f"wrd_pricetaker_summary_results_{timestamp}.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"Saved summary results to: {results_csv}")
