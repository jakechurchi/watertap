import warnings
import logging

warnings.filterwarnings("ignore", message=".*implicit domain of 'Any'.*")
logging.getLogger("pyomo").setLevel(logging.ERROR)
from idaes.apps.grid_integration import PriceTakerModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints
from pathlib import Path

from watertap.flowsheets.flex_desal import wrd_ro_flowsheet as fs
from watertap.flowsheets.flex_desal import utils
from watertap.flowsheets.flex_desal.params import FlexDesalParams
from watertap.core.solvers import get_solver

from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.core.util.model_statistics import degrees_of_freedom


def plot_function(m, n_time_points):
    time = np.linspace(0, n_time_points - 1, n_time_points)
    fig, (ax_energy, ax_demand, ax_trains) = plt.subplots(3, 1, figsize=(12, 12))

    # First subplot: Energy consumption and electricity price
    energy = [
        pyo.value(v[None])
        for v in m.period[:, :].net_power_consumption.extract_values()
    ]
    ax_energy.plot(
        time + 0.5, energy, label="Energy Consumption (kWh)", color="orange", marker="o"
    )
    ax_energy.set_ylim(0, 2500)
    ax_energy.set_ylabel("Energy Consumption (kWh)", fontsize=12)
    ax_energy.set_title(
        "Energy Consumption and Electricity Price", fontsize=14, fontweight="bold"
    )
    ax_energy.grid(False)

    ax_price = ax_energy.twinx()
    elec_price = m._config.lmp_data
    ax_price.plot(
        time + 0.5,
        elec_price,
        label="Electricity Cost ($/kWh)",
        color="black",
        linestyle="-",
        linewidth=2,
    )
    ax_price.set_ylabel("Electricity Cost ($/kWh)", fontsize=12)
    ax_price.set_ylim(0, 0.17)

    # Add working hours shading
    for i in range(int(n_time_points / 24)):
        ax_energy.axvspan(
            24 * i,
            24 * i + 8,
            facecolor="grey",
            alpha=0.1,
            label="Nonworking Hours" if i == 0 else "_nolegend_",
            zorder=0,
        )
        ax_energy.axvspan(
            24 * i + 8,
            24 * i + 18,
            facecolor="gold",
            alpha=0.3,
            label="Working Hours" if i == 0 else "_nolegend_",
            zorder=0,
        )
        ax_energy.axvspan(
            24 * i + 18,
            24 * i + 24,
            facecolor="grey",
            alpha=0.1,
            label="_nolegend_",
            zorder=0,
        )

    handle1, label1 = ax_energy.get_legend_handles_labels()
    handle2, label2 = ax_price.get_legend_handles_labels()
    handles = handle2 + handle1
    labels = label2 + label1
    leg1 = ax_price.legend(
        handles, labels, loc="lower left", framealpha=1.0, ncol=2, fontsize=11
    )
    leg1.set_zorder(1000)
    leg1.get_frame().set_facecolor("white")
    ax_energy.xaxis.set_major_locator(plt.MaxNLocator(24))

    # Second subplot: Demand charges
    fixed_demand_profile = [
        v[None] for v in m.period[:, :].fixed_demand_rate.extract_values()
    ]
    fixed_line = ax_demand.plot(
        time + 0.5,
        fixed_demand_profile,
        label="Fixed Demand Charge",
        color="red",
        linestyle="-",
        linewidth=2,
    )
    on_peak_demand_profile = [
        v[None] for v in m.period[:, :].variable_demand_rate.extract_values()
    ]
    on_peak_line = ax_demand.plot(
        time + 0.5,
        on_peak_demand_profile,
        label="On-Peak Demand Charge",
        color="purple",
        linestyle="-",
        linewidth=2,
    )

    for i in range(int(n_time_points)):
        ax_demand.axvspan(
            24 * i,
            24 * i + 8,
            facecolor="grey",
            alpha=0.1,
            label="Nonworking Hours" if i == 0 else "_nolegend_",
            zorder=0,
        )
        ax_demand.axvspan(
            24 * i + 8,
            24 * i + 18,
            facecolor="gold",
            alpha=0.3,
            label="Working Hours" if i == 0 else "_nolegend_",
            zorder=0,
        )
        ax_demand.axvspan(
            24 * i + 18,
            24 * i + 24,
            facecolor="grey",
            alpha=0.1,
            label="_nolegend_",
            zorder=0,
        )

    ax_demand.legend(
        handles=[fixed_line[0], on_peak_line[0]],
        loc="lower left",
        fontsize=11,
        framealpha=1.0,
    )
    ax_demand.xaxis.set_major_locator(plt.MaxNLocator(24))
    ax_demand.set_ylabel("Demand Charge ($/kW)", fontsize=12)
    ax_demand.set_title("Demand Charges", fontsize=14, fontweight="bold")

    # Third subplot: Water production and RO train flow rates
    prod = [
        v[None] for v in m.period[:, :].posttreatment.product_flowrate.extract_values()
    ]
    ax_trains.plot(
        time + 0.5, prod, label="Water Production (m3/hr)", color="blue", marker="o"
    )
    ax_trains.set_ylim(0, 2500)
    ax_trains.axhline(
        y=53150 / 24,
        color="blue",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label="Maximum Production Capacity (m3/h)",
        zorder=0,
    )
    ax_trains.set_ylabel("Water Production (m3/h)", fontsize=12)
    ax_trains.set_xlabel("Hours", fontsize=12)
    ax_trains.set_title(
        "Water Production & RO Train Flow Rates", fontsize=14, fontweight="bold"
    )
    ax_trains.xaxis.set_major_locator(plt.MaxNLocator(24))
    ax_trains.grid(False)

    # Extract RO train flow rates and convert to % of max flow
    max_train_flow = 53150 / 24 / 4  # m3/hr
    train_1_flows = [
        v[None] / max_train_flow * 100
        for v in m.period[:, :]
        .reverse_osmosis.ro_skid[1]
        .product_flowrate.extract_values()
    ]
    train_2_flows = [
        v[None] / max_train_flow * 100
        for v in m.period[:, :]
        .reverse_osmosis.ro_skid[2]
        .product_flowrate.extract_values()
    ]
    train_3_flows = [
        v[None] / max_train_flow * 100
        for v in m.period[:, :]
        .reverse_osmosis.ro_skid[3]
        .product_flowrate.extract_values()
    ]
    train_4_flows = [
        v[None] / max_train_flow * 100
        for v in m.period[:, :]
        .reverse_osmosis.ro_skid[4]
        .product_flowrate.extract_values()
    ]

    ax_trains_pct = ax_trains.twinx()
    ax_trains_pct.plot(
        time + 0.5, train_1_flows, label="RO Train 1", marker="o", linewidth=2
    )
    ax_trains_pct.plot(
        time + 0.5, train_2_flows, label="RO Train 2", marker="s", linewidth=2
    )
    ax_trains_pct.plot(
        time + 0.5, train_3_flows, label="RO Train 3", marker="^", linewidth=2
    )
    ax_trains_pct.plot(
        time + 0.5, train_4_flows, label="RO Train 4", marker="d", linewidth=2
    )
    ax_trains_pct.axhline(
        y=100,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label="Max Capacity",
        zorder=0,
    )
    ax_trains_pct.set_ylim(0, 125)
    ax_trains_pct.set_ylabel("Flow Rate (% of Max)", fontsize=12)

    for i in range(int(n_time_points / 24)):
        ax_trains_pct.axvspan(
            24 * i,
            24 * i + 8,
            facecolor="grey",
            alpha=0.1,
            label="Nonworking Hours" if i == 0 else "_nolegend_",
            zorder=0,
        )
        ax_trains_pct.axvspan(
            24 * i + 8,
            24 * i + 18,
            facecolor="gold",
            alpha=0.3,
            label="Working Hours" if i == 0 else "_nolegend_",
            zorder=0,
        )
        ax_trains_pct.axvspan(
            24 * i + 18,
            24 * i + 24,
            facecolor="grey",
            alpha=0.1,
            label="_nolegend_",
            zorder=0,
        )

    handle_t, label_t = ax_trains.get_legend_handles_labels()
    handle_t2, label_t2 = ax_trains_pct.get_legend_handles_labels()
    leg3 = ax_trains_pct.legend(
        handle_t + handle_t2,
        label_t + label_t2,
        loc="lower left",
        fontsize=11,
        framealpha=1.0,
        ncol=3,
    )
    leg3.get_frame().set_facecolor("white")

    # Set consistent x-axis limits and formatting
    for a in (ax_energy, ax_demand, ax_trains):
        a.set_xlim(0, n_time_points)
        a.xaxis.set_major_locator(plt.MaxNLocator(24))

    # Tick labels for all axes
    for a in (ax_energy, ax_price, ax_demand, ax_trains, ax_trains_pct):
        a.tick_params(axis="both", labelsize=11)

    fig.tight_layout()
    fig.savefig("wrd_pricetaker_summer_month.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    # Load price data
    price_data = pd.read_csv(script_dir / "wrd_pricesignal_summer_week.csv")
    price_data["Energy Rate"] = (
        price_data["electric_energy_on_peak"]
        + price_data["electric_energy_mid_peak"]
        + price_data["electric_energy_off_peak"]
        + price_data["electric_energy_super_off_peak"]
    )
    price_data["Fixed Demand Rate"] = price_data["electric_demand_fixed"]
    price_data["Var Demand Rate"] = price_data["electric_demand_peak"]
    price_data["Customer Cost"] = price_data["electric_customer_fixed_charge"]

    # price_data["Energy Rate"] = (
    #     price_data["electric_energy_0_2022-07-05_2022-07-14_0"]
    #     + price_data["electric_energy_1_2022-07-05_2022-07-14_0"]
    #     + price_data["electric_energy_2_2022-07-05_2022-07-14_0"]
    #     + price_data["electric_energy_3_2022-07-05_2022-07-14_0"]
    # )
    # price_data["Fixed Demand Rate"] = price_data[
    #     "electric_demand_maximum_2022-07-05_2022-07-14_0"
    # ]
    # price_data["Var Demand Rate"] = price_data[
    #     "electric_demand_peak-summer_2022-07-05_2022-07-14_0"
    # ]
    # price_data["Customer Cost"] = price_data[
    #     "electric_customer_0_2022-07-05_2022-07-14_0"
    # ]

    price_data["Emissions Intensity"] = 0

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
        include_onsite_solar=True,
        onsite_capacity=pv_capacity,
        nonworking_hours=list(range(0, 8))
        + list(
            range(18, 24)
        ),  # 6pm-8am are nonworking hours (assuming time index starts at 0 for 12am-1am)
        # rainy_days=1,
        CAPEX_yr=6498300,  # For WRD, this assumes a 30 yr lifetime
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
            "surrogate_a": 2.83e-1,
            "surrogate_b": -3.44e-4,
            "surrogate_c": 2.46e-7,
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
            "surrogate_type": "quadratic_energy_intensity",
            "surrogate_a": 5.411e-1,
            "surrogate_b": -9.826e-4,
            "surrogate_c": 1.100e-6,
            "nominal_recovery": 0.92,
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
        }  # This number is not confirmed at all
    )  # kWh/m3 #$/m3

    m.params.brinedischarge.update({"brine_cost": 0.43, "energy_intensity": 0})

    # Append LMP data to the model
    m.append_lmp_data(lmp_data=price_data["Energy Rate"])

    m.build_multiperiod_model(
        flowsheet_func=fs.build_desal_flowsheet,
        flowsheet_options={"params": m.params},
    )

    # Update the time-varying parameters other than the LMP, such as
    # demand costs and emissions intensity. LMP value is updated by default
    m.update_operation_params(
        {
            "fixed_demand_rate": price_data["Fixed Demand Rate"],
            "variable_demand_rate": price_data["Var Demand Rate"],
            "emissions_intensity": price_data["Emissions Intensity"],
            "customer_cost": price_data["Customer Cost"],
            "power_generation.capacity_factor": pv_capacity_factors,
        }
    )

    # Add demand cost and fixed cost calculation constraints
    fs.add_demand_and_fixed_costs(m)

    # Add the startup delay constraints
    fs.add_delayed_startup_constraints(m)

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
    fs.add_replacement_costs_piecewise(m)

    m.total_op_cost = pyo.Expression(
        expr=m.total_energy_cost
        + m.total_demand_cost
        + m.total_customer_cost
        + m.total_feed_cost
        + m.total_brine_cost
        + m.total_chemical_cost
        + m.total_replacement_cost  # function of degree of flexibility
    )
    # add CAPEX as a fixed cost to calculate LCOW
    m.fixed_cost = pyo.Expression(expr=m.params.CAPEX_yr * m.params.num_months / 12)
    m.total_cost = pyo.Expression(expr=m.total_op_cost + m.fixed_cost)

    m.LCOW = pyo.Expression(expr=m.total_cost / m.total_water_production)  # $/m3

    fs.constrain_water_production(m)

    # If water recovery is static, it must be fixed
    if not m.params.wrd_ro.allow_variable_recovery:
        utils.wrd_fix_recovery(
            m,
            ro_recovery=m.params.wrd_ro.nominal_recovery,
            uf_recovery=m.params.wrd_uf.nominal_recovery,
        )

    # Could cause feasibility issues b/c this is a slakc varable essentially.
    # m.fix_operation_var("reverse_osmosis.leftover_flow", 0)

    # Flowrates not fixed, but shouldn't randomly fluctuate either.
    fs.add_flow_changes_penalty_binary(m)

    # fs.add_working_hours_constraint(m)

    # fs.add_rain_shutdowns(m)

    # To define a baseline
    m.obj = pyo.Objective(
        expr=1e-4 * (m.total_op_cost + m.flow_changes_penalty),
        sense=pyo.minimize,
    )

    # Only to find the baseline power for this water production
    # m.enforce_steady_state = pyo.Constraint(expr=m.flow_changes_penalty == 0)

    print(degrees_of_freedom(m))

    # dt = DiagnosticsToolbox(m)
    # dt.report_structural_issues()
    # solver = get_solver()
    # solver.options["max_iter"] = 500
    # results = solver.solve(m, tee=True)

    mip_gap = 0.01
    solver = pyo.SolverFactory("gurobi_direct_minlp")
    solver.options["MIPGap"] = mip_gap
    # solver.options["MIPFocus"] = 2
    # solver.options["StartNodeLimit"] = (
    #     50000  # I think this will allow it to complete the partial solution I'm initializing above.
    # )
    results = solver.solve(m, tee=True)

    print(f"m.flow_changes_penalty(): {m.flow_changes_penalty()}")
    print(f"Total operational cost: {m.total_op_cost():.2f}")

    # termination_condition = results.solver.termination_condition
    # print(f"Solver termination condition: {termination_condition}")

    # if termination_condition in (
    #     pyo.TerminationCondition.infeasible,
    #     pyo.TerminationCondition.infeasibleOrUnbounded,
    # ):
    #     print("\nModel is infeasible. Logging infeasible constraints...")
    #     logging.getLogger("pyomo.util.infeasible").setLevel(logging.INFO)
    #     log_infeasible_constraints(
    #         m,
    #         tol=1e-7,
    #         log_expression=True,
    #         log_variables=True,
    #     )
    #     raise RuntimeError("Model infeasible. See logs above for violated constraints.")

    pyo.assert_optimal_termination(results)

    # Baseline power is a function of the target water production, but needs to be calculated by running this model!
    fs.calculate_flexibility_metrics(m, baseline_power=1000)

    design_var_values = m.get_design_var_values()
    filtered_design_var_values = {
        k: v
        for k, v in design_var_values.items()
        if "flow_change" not in k and "flow_changed" not in k and "reduction" not in k
    }
    print(filtered_design_var_values)

    # Write optimal values of all operational variables to a csv file
    output_csv = script_dir / "wrd_dummy_result.csv"
    m.get_operation_var_values().to_csv(output_csv)
    print(f"Saved operation variable results to: {output_csv}")

    plot_function(m, n_time_points=len(price_data))

    # Plot operational variables
    fig, axs = m.plot_operation_profile(
        operation_vars=[
            "fixed_demand_rate",
            "variable_demand_rate",
            "posttreatment.product_flowrate",
            "num_skids_online",
        ],
    )
    fig.savefig("wrd_operation_profile.png")
