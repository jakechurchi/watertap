import warnings
import logging

warnings.filterwarnings("ignore", message=".*implicit domain of 'Any'.*")
logging.getLogger("pyomo").setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime

import pyomo.environ as pyo
from pyomo.environ import SolverFactory, value

from watertap.flowsheets.flex_desal import wrd_ro_flowsheet as fs
from watertap.flowsheets.flex_desal import utils
from watertap.flowsheets.flex_desal.params import FlexDesalParams
from watertap.core.solvers import get_solver

from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.apps.grid_integration import PriceTakerModel


def plot_function(m, n_time_points, output_stem, peak_hours=None):
    time = np.linspace(0, n_time_points - 1, n_time_points)
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
    ax_energy = fig.add_subplot(gs[0])
    ax_trains = fig.add_subplot(gs[1], sharex=ax_energy)
    ax_energy.set_facecolor("#f5f5f5")
    ax_trains.set_facecolor("#f5f5f5")

    if peak_hours is not None:
        peak_legend_added = False
        for i, is_peak in enumerate(peak_hours):
            if is_peak:
                # Shade full hourly intervals where variable demand charges apply.
                span_label = "Peak Hours" if not peak_legend_added else None
                ax_energy.axvspan(
                    i,
                    i + 1,
                    color="grey",
                    alpha=0.2,
                    linewidth=0,
                    zorder=-1,
                    hatch="///",
                    label=span_label,
                )
                ax_trains.axvspan(
                    i,
                    i + 1,
                    color="grey",
                    alpha=0.2,
                    linewidth=0,
                    zorder=-1,
                    hatch="///",
                    label=span_label,
                )
                peak_legend_added = True

    # First subplot: Stacked energy consumption by major equipment
    total_energy = [
        pyo.value(v[None])
        for v in m.period[:, :].net_power_consumption.extract_values()
    ]

    ro1_energy = [
        v[None]
        for v in m.period[:, :]
        .reverse_osmosis.ro_skid[1]
        .power_consumption.extract_values()
    ]
    ro2_energy = [
        v[None]
        for v in m.period[:, :]
        .reverse_osmosis.ro_skid[2]
        .power_consumption.extract_values()
    ]
    ro3_energy = [
        v[None]
        for v in m.period[:, :]
        .reverse_osmosis.ro_skid[3]
        .power_consumption.extract_values()
    ]
    ro4_energy = [
        v[None]
        for v in m.period[:, :]
        .reverse_osmosis.ro_skid[4]
        .power_consumption.extract_values()
    ]

    uf1_energy = [
        v[None]
        for v in m.period[:, :]
        .pretreatment.uf_pumps[1]
        .power_consumption.extract_values()
    ]
    uf2_energy = [
        v[None]
        for v in m.period[:, :]
        .pretreatment.uf_pumps[2]
        .power_consumption.extract_values()
    ]
    uf3_energy = [
        v[None]
        for v in m.period[:, :]
        .pretreatment.uf_pumps[3]
        .power_consumption.extract_values()
    ]

    other_energy = np.array(total_energy) - (
        np.array(ro1_energy)
        + np.array(ro2_energy)
        + np.array(ro3_energy)
        + np.array(ro4_energy)
        + np.array(uf1_energy)
        + np.array(uf2_energy)
        + np.array(uf3_energy)
    )
    # Clip tiny negatives from solver tolerances so stackplot remains well-defined.
    other_energy = np.maximum(other_energy, 0.0)

    ax_energy.stackplot(
        time + 0.5,
        ro1_energy,
        ro2_energy,
        ro3_energy,
        ro4_energy,
        uf1_energy,
        uf2_energy,
        uf3_energy,
        other_energy,
        labels=[
            "RO Train 1",
            "RO Train 2",
            "RO Train 3",
            "RO Train 4",
            "UF Pump 1",
            "UF Pump 2",
            "UF Pump 3",
            "Post-Treatment",
        ],
        alpha=0.5,
    )

    ax_energy.plot(
        time + 0.5,
        total_energy,
        label="Total Power",
        color="black",
        linestyle="--",
        linewidth=2,
    )
    ax_energy.set_ylim(0, 2500)
    ax_energy.set_ylabel("kW", fontsize=12)
    ax_energy.grid(False)

    ax_price = ax_energy.twinx()
    elec_price = np.asarray(m._config.lmp_data, dtype=float)
    if elec_price.size != n_time_points:
        if elec_price.size % n_time_points == 0:
            elec_price = elec_price.reshape(n_time_points, -1).mean(axis=1)
        else:
            elec_price = elec_price[:n_time_points]
    ax_price.plot(
        time + 0.5,
        elec_price * 100,
        label="Elec. Price",
        color="orange",
        linewidth=2,
    )
    ax_price.set_ylabel("¢/kWh", fontsize=12)
    ax_price.set_ylim(0, max(elec_price * 100) + 3)

    handle1, label1 = ax_energy.get_legend_handles_labels()
    handle2, label2 = ax_price.get_legend_handles_labels()
    handles = handle2 + handle1
    labels = label2 + label1
    leg1 = ax_price.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.0, 1, 1, 1),
        framealpha=1.0,
        ncol=4,
        fontsize=12,
        mode="expand",
    )
    leg1.set_zorder(1000)
    leg1.get_frame().set_facecolor("white")
    ax_energy.xaxis.set_major_locator(plt.MaxNLocator(24))

    # Second subplot: Water production and RO train flow rates
    prod = [
        v[None] for v in m.period[:, :].posttreatment.product_flowrate.extract_values()
    ]
    ax_trains.plot(
        time + 0.5,
        prod,
        label="Water Production",
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.75,
    )
    ax_trains.set_ylim(0, 2500)
    ax_trains.axhline(
        y=635 * 0.925 * 4,
        color="blue",
        linestyle=":",
        linewidth=2,
        alpha=0.75,
        label="Max Production",
        zorder=0,
    )
    ax_trains.set_ylabel("m$^3$/h", fontsize=12)
    ax_trains.set_xlabel("Hours", fontsize=12)
    ax_trains.xaxis.set_major_locator(plt.MaxNLocator(24))
    ax_trains.grid(False)

    # Extract RO train flow rates (m3/hr) for stacked plotting
    train_1_flows = [
        v[None]
        for v in m.period[:, :]
        .reverse_osmosis.ro_skid[1]
        .product_flowrate.extract_values()
    ]
    train_2_flows = [
        v[None]
        for v in m.period[:, :]
        .reverse_osmosis.ro_skid[2]
        .product_flowrate.extract_values()
    ]
    train_3_flows = [
        v[None]
        for v in m.period[:, :]
        .reverse_osmosis.ro_skid[3]
        .product_flowrate.extract_values()
    ]
    train_4_flows = [
        v[None]
        for v in m.period[:, :]
        .reverse_osmosis.ro_skid[4]
        .product_flowrate.extract_values()
    ]

    ax_trains.stackplot(
        time + 0.5,
        train_1_flows,
        train_2_flows,
        train_3_flows,
        train_4_flows,
        labels=["RO Train 1", "RO Train 2", "RO Train 3", "RO Train 4"],
        alpha=0.5,
    )

    handle_t, label_t = ax_trains.get_legend_handles_labels()
    leg3 = ax_trains.legend(
        handle_t,
        label_t,
        loc="lower left",
        bbox_to_anchor=(0.0, 1, 1, 1),
        framealpha=1.0,
        ncol=3,
        fontsize=12,
        mode="expand",
    )
    leg3.get_frame().set_facecolor("white")

    # Set consistent x-axis limits and formatting
    for a in (ax_energy, ax_trains):
        a.set_xlim(0, n_time_points)
        a.xaxis.set_major_locator(plt.MaxNLocator(24))

    # Tick labels for all axes
    for a in (ax_energy, ax_price, ax_trains):
        a.tick_params(axis="both", labelsize=14)
    for label in ax_trains.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("center")
    for label in ax_energy.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("center")

    fig.tight_layout()
    fig.savefig(f"{output_stem}.png", dpi=600)
    # plt.show()


def _begin_and_end_constraint(m):
    """Force RO train 1 op_mode to match between first and last timesteps."""
    period_points = list(m.period.index_set())
    if not period_points:
        return

    first_point = period_points[0]
    last_point = period_points[-1]

    @m.Constraint()
    def match_train_1_at_start_and_end(blk):
        return (
            blk.period[first_point].reverse_osmosis.ro_skid[1].op_mode
            == blk.period[last_point].reverse_osmosis.ro_skid[1].op_mode
        )


def one_week(
    annual_production_AF=13000, flex_type=None, season="summer", num_shutdowns=None
):

    season_map = {
        "summer": "price_signals/summer_week.csv",
        "winter": "price_signals/winter_week.csv",
    }
    season_key = season.lower()
    if season_key not in season_map:
        raise ValueError(
            f"Invalid season '{season}'. Valid options are: {sorted(season_map)}"
        )

    flex_type_key = flex_type.lower()
    valid_flex_types = {"both", "no_flex", "num_shutdowns"}
    if flex_type_key not in valid_flex_types:
        raise ValueError(
            "Invalid flex_type "
            f"'{flex_type}'. Valid options are: {sorted(valid_flex_types)}"
        )
    selected_price_signal_stem = Path(season_map[season_key]).stem
    if flex_type_key == "num_shutdowns":
        output_suffix = (
            f"{flex_type_key}_{num_shutdowns}_{season_key}_{annual_production_AF}AF"
        )
    else:
        output_suffix = f"{flex_type_key}_{season_key}_{annual_production_AF}AF"
    if selected_price_signal_stem.upper().endswith("RTP"):
        output_suffix = f"{output_suffix}_RTP"
    if selected_price_signal_stem.upper().endswith("TOU_8"):
        output_suffix = f"{output_suffix}_TOU_8"
    if selected_price_signal_stem.upper().endswith("CPP"):
        output_suffix = f"{output_suffix}_CPP"
    if selected_price_signal_stem.upper().endswith("DR"):
        output_suffix = f"{output_suffix}_DR"

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
        annual_production_AF=annual_production_AF,
        timestep_hours=timestep_hours,
        CAPEX_yr=6498300,  # For WRD, this assumes a 30 yr lifetime
        include_demand_response=True,
    )
    m.baseline_power = 1102  # kW
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
                125000 * 4,
            ],  # $ per replacement
            "replacement_lifetimes": [5, 17.5],  # years
            "replacement_max_flex_penalty": [
                0.1,
                0.1,
            ],  # Reduction in lifetime if shutdowns occur twice a day
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

    _begin_and_end_constraint(m)

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

    # Add demand cost and fixed cost calculation constraints
    fs.add_demand_and_fixed_costs(m)

    # Add the startup delay constraints
    fs.add_delayed_startup_constraints(m)
    fs.add_delayed_shutdown_constraints(m)

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

    # Could cause feasibility issues b/c this is a slack variable essentially.
    # m.fix_operation_var("reverse_osmosis.leftover_flow", 0)

    # Flowrates not fixed, but shouldn't randomly fluctuate either.
    fs.add_flow_changes_penalty_binary(m)

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

    # m.obj = pyo.Objective(expr = m.total_water_production, sense=pyo.maximize)

    # Only to find the baseline power for this water production
    if flex_type_key == "no_flex":
        m.enforce_steady_state = pyo.Constraint(expr=m.flow_changes_penalty == 0)

        # ADDING A MAXIMUM FLOW CASE
        @m.Constraint(range(1, m.params.wrd_ro.num_ro_skids + 1))
        def max_flow_constraint(m_blk, i):
            return (
                m.period[1, 1].reverse_osmosis.ro_skid[i].feed_flowrate
                >= m.params.wrd_ro.maximum_flowrate
            )

    if flex_type_key == "num_shutdowns":
        # Add constraint to allow for exactly num_shutdowns during the period for each RO train.
        @m.Constraint(range(1, m.params.wrd_ro.num_ro_skids + 1))
        def num_shutdowns_constraint(m_blk, i):
            return (
                sum(
                    m.period[d, t].reverse_osmosis.ro_skid[i].shutdown
                    for d in m.set_days
                    for t in m.set_time
                )
                <= num_shutdowns
            )

        # Add constraint to allow for exactly num_startups during the period for each RO train.
        @m.Constraint(range(1, m.params.wrd_ro.num_ro_skids + 1))
        def num_startups_constraint(m_blk, i):
            return (
                sum(
                    m.period[d, t].reverse_osmosis.ro_skid[i].startup
                    for d in m.set_days
                    for t in m.set_time
                )
                <= num_shutdowns
            )

    print(degrees_of_freedom(m))

    # dt = DiagnosticsToolbox(m)
    # dt.report_structural_issues()

    # IPOPT
    # solver = get_solver()
    # solver.options["max_iter"] = 500

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
    # The OPEX value does not include the replacement costs... so I guess they aren't being included in the LVOF
    if season_key == "winter":
        if selected_price_signal_stem.upper().endswith("TOU_8"):
            baseline_OPEX = 115031
        elif selected_price_signal_stem.upper().endswith("RTP"):
            baseline_OPEX = 116862
        else:
            baseline_OPEX = 111145  # $
    else:
        if selected_price_signal_stem.upper().endswith("TOU_8"):
            baseline_OPEX = 124771
        elif selected_price_signal_stem.upper().endswith("RTP"):
            baseline_OPEX = 187020
        elif selected_price_signal_stem.upper().endswith("CPP"):
            baseline_OPEX = 123683
        else:
            baseline_OPEX = 120098  # $

    fs.calculate_replacement_costs(m)
    fs.calculate_flexibility_metrics(
        m,
        baseline_power=value(
            m.baseline_power
        ),  # kW, from the baseline with steady production and 12000 AF/yr water production
        baseline_OPEX=baseline_OPEX,
    )

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

    return filtered_design_var_values


if __name__ == "__main__":
    # Inputs
    water_prod_targs = [
        1
    ]  # mostly to compare to the results I already have tabulated to see if they've changed at all
    season = "winter"
    flex_type = "no_flex"
    number_of_shutdowns = [10]  # 10 is essentially unlimited shutdowns allowed

    # Outputs
    water = []
    cost = []
    energy_cost = []
    demand_cost = []
    feed_cost = []
    brine_cost = []
    chemical_cost = []
    replacement_cost = []
    deg_of_flex = []
    electricity_cost = []
    LCOW = []
    annual_production_values = []
    allowed_shutdown_values = []

    for annual_production in water_prod_targs:
        for i in number_of_shutdowns:
            print(
                f"\n\nRunning optimization for annual production of {annual_production} AF..."
            )
            design_vars = one_week(
                annual_production_AF=annual_production,
                flex_type=flex_type,
                season=season,
                num_shutdowns=i,
            )
            water.append(design_vars["total_water_production"])
            cost.append(design_vars["total_cost"])
            energy_cost.append(design_vars["total_energy_cost"])
            demand_cost.append(design_vars["total_demand_cost"])
            feed_cost.append(design_vars["total_feed_cost"])
            brine_cost.append(design_vars["total_brine_cost"])
            chemical_cost.append(design_vars["total_chemical_cost"])
            replacement_cost.append(design_vars["total_replacement_cost"])
            deg_of_flex.append(design_vars["degree_of_flex"])
            electricity_cost.append(
                design_vars["total_demand_cost"] + design_vars["total_energy_cost"]
            )
            LCOW.append(design_vars["LCOW"])
            annual_production_values.append(annual_production)
            allowed_shutdown_values.append(i)

    df = pd.DataFrame(
        {
            "Annual Production (AF)": annual_production_values,
            "Number of Allowed Shutdowns": allowed_shutdown_values,
            "Total Water Production (m3)": water,
            "Total Cost ($)": cost,
            "Total Energy Cost ($)": energy_cost,
            "Total Demand Cost ($)": demand_cost,
            "Total Feed Cost ($)": feed_cost,
            "Total Brine Cost ($)": brine_cost,
            "Total Chemical Cost ($)": chemical_cost,
            "Total Electricity Cost ($)": electricity_cost,
            "Total Replacement Cost ($)": replacement_cost,
            "Degree of Flexibility": deg_of_flex,
            "Levelized Cost of Water ($/m3)": LCOW,
        }
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(
        f"water_targ_sweep_week_{season}_{flex_type}_{timestamp}.csv", index=False
    )
