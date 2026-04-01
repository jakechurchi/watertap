import warnings
import logging

warnings.filterwarnings("ignore", message=".*implicit domain of 'Any'.*")
logging.getLogger("pyomo").setLevel(logging.ERROR)
from idaes.apps.grid_integration import PriceTakerModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyomo.environ as pyo
from pathlib import Path
from watertap.flowsheets.flex_desal import wrd_ro_flowsheet as fs
from watertap.flowsheets.flex_desal import utils
from watertap.flowsheets.flex_desal.params import FlexDesalParams
from watertap.core.solvers import get_solver


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
    price_data = pd.read_csv(script_dir / "sbce_pricesignal_short.csv")
    # price_data["Energy Rate"] = (
    #     price_data["electric_energy_on_peak"]
    #     + price_data["electric_energy_mid_peak"]
    #     + price_data["electric_energy_off_peak"]
    #     + price_data["electric_energy_super_off_peak"]
    # )
    # price_data["Fixed Demand Rate"] = price_data["electric_demand_fixed_summer"]
    # price_data["Var Demand Rate"] = price_data["electric_demand_peak_summer"]
    price_data["Energy Rate"] = (
        price_data["electric_energy_0_2022-07-05_2022-07-14_0"]
        + price_data["electric_energy_1_2022-07-05_2022-07-14_0"]
        + price_data["electric_energy_2_2022-07-05_2022-07-14_0"]
        + price_data["electric_energy_3_2022-07-05_2022-07-14_0"]
    )
    price_data["Fixed Demand Rate"] = price_data[
        "electric_demand_maximum_2022-07-05_2022-07-14_0"
    ]
    price_data["Var Demand Rate"] = price_data[
        "electric_demand_peak-summer_2022-07-05_2022-07-14_0"
    ]
    price_data["Emissions Intensity"] = 0
    # price_data["Customer Cost"] = price_data["electric_customer_fixed_charge"]
    price_data["Customer Cost"] = price_data[
        "electric_customer_0_2022-07-05_2022-07-14_0"
    ]
    m = PriceTakerModel()

    # Instantiate an object containing the model parameters
    m.params = FlexDesalParams(
        start_date="2022-07-05 00:00:00",
        end_date="2022-07-05 02:15:00",
        annual_production_AF=11800,
        timestep_hours=0.25,
        # fixed_monthly_cost = 10000,
        # customer_rate=price_data["Customer Cost"][1],  # acrft/yr
    )
    # Add a check that the dates match the price data

    m.params.intake.update({"energy_intensity": 0, "nominal_flowrate": 2500})  # m3/hr
    # m.params.pretreatment.update({"energy_intensity": 0})
    m.params.wrd_uf.update(
        {
            "minimum_flowrate": 344,  # m3/hr
            "nominal_flowrate": 900,
            "maximum_flowrate": 989,
            "surrogate_type": "quadratic_energy_intensity",
            "surrogate_a": 2.83e-1,
            "surrogate_b": -3.44e-4,
            "surrogate_c": 2.46e-7,
            "nominal_recovery": 0.96,
        }
    )

    m.params.wrd_ro.update(
        {
            "startup_delay": 3,  # hours
            "minimum_downtime": 3,  # hours
            "minimum_flowrate": 520,  # m3/hr
            "nominal_flowrate": 602,
            "maximum_flowrate": 635,
            "surrogate_type": "quadratic_energy_intensity",
            "surrogate_a": 7.83e-1,
            "surrogate_b": 1.588e-3,
            "surrogate_c": 1.673e-6,
            "nominal_recovery": 0.92,
            "num_ro_skids": 4,
        }
    )

    m.params.posttreatment.update({"energy_intensity": 0})

    # Append LMP data to the model
    m.append_lmp_data(lmp_data=price_data["Energy Rate"])

    m.build_multiperiod_model(
        flowsheet_func=fs.build_desal_flowsheet,
        flowsheet_options={"params": m.params},
    )

    # Update the time-varying parameters other than the LMP, such as
    # demand costs and emissions intensity. LMP value is updated by default

    # First, discover what blocks exist in the model
    print("Discovering blocks in period[1,1]:")

    for component in m.period[1, 1].component_objects(pyo.Block, descend_into=False):
        print(f"  - {component.name}")

    m.update_operation_params(
        {
            "fixed_demand_rate": price_data["Fixed Demand Rate"],
            "variable_demand_rate": price_data["Var Demand Rate"],
            "emissions_intensity": price_data["Emissions Intensity"],
            "customer_cost": price_data["Customer Cost"],
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
    m.total_electricity_cost = pyo.Expression(
        expr=m.total_energy_cost + m.total_demand_cost + m.total_customer_cost
    )

    # Feed flow to the intake does not vary with time
    m.fix_operation_var("intake.feed_flowrate", m.params.intake.nominal_flowrate)

    # Pretreatment is either active (1) or inactive (0) for the entire run
    m.fix_operation_var("pretreatment.op_mode", 1)

    fs.constrain_water_production(m)

    # If water recovery is static, it must be fixed
    if not m.params.wrd_ro.allow_variable_recovery:
        utils.wrd_fix_recovery(m, recovery=m.params.wrd_ro.nominal_recovery)

    m.num_shutdowns = pyo.Expression(
        expr=1
        * sum(
            sum(m.period[:, :].reverse_osmosis.ro_skid[i].shutdown)
            for i in range(1, m.params.wrd_ro.num_ro_skids + 1)
        )
    )  # 5 is an abitrary scaling factor

    # Add binary variables to track flowrate changes between consecutive periods
    m.flow_changed = pyo.Var(
        m.set_days,
        m.set_time,
        range(1, m.params.wrd_ro.num_ro_skids + 1),
        within=pyo.Binary,
        doc="Binary variable: 1 if flowrate changes from previous period, 0 otherwise",
    )

    # Add constraints to detect flowrate changes
    # We need a big-M value for the constraint (use maximum flowrate as big-M)
    big_M = m.params.wrd_ro.maximum_flowrate * 2

    @m.Constraint(m.set_days, m.set_time, range(1, m.params.wrd_ro.num_ro_skids + 1))
    def track_flow_changes(m_blk, d, t, i):
        # Skip first time period of first day (no previous period to compare)
        if d == 1 and t == 1:
            return pyo.Constraint.Skip

        # Get current and previous period flowrates
        current_flow = m_blk.period[d, t].reverse_osmosis.ro_skid[i].feed_flowrate

        if t == 1:
            # First hour of a day (not first day), compare to last hour of previous day
            prev_flow = (
                m_blk.period[d - 1, m_blk.set_time.last()]
                .reverse_osmosis.ro_skid[i]
                .feed_flowrate
            )
        else:
            # Compare to previous hour in same day
            prev_flow = m_blk.period[d, t - 1].reverse_osmosis.ro_skid[i].feed_flowrate

        # If flows are different, flow_changed must be 1
        # This constraint allows flow_changed to be 1 when |current - prev| > 0
        # Using a tolerance-based approach: if difference > small threshold, binary = 1
        flow_diff = current_flow - prev_flow

        # Note: This is a simplified constraint that encourages flow_changed = 1 when flow differs
        # but doesn't strictly enforce it. For strict enforcement, you'd need indicator constraints
        # or absolute value formulation which is more complex.
        # Since we're minimizing, the solver will naturally set flow_changed = 0 when possible
        return m_blk.flow_changed[d, t, i] * big_M >= flow_diff

    @m.Constraint(m.set_days, m.set_time, range(1, m.params.wrd_ro.num_ro_skids + 1))
    def track_flow_changes_neg(m_blk, d, t, i):
        # Skip first time period of first day
        if d == 1 and t == 1:
            return pyo.Constraint.Skip

        current_flow = m_blk.period[d, t].reverse_osmosis.ro_skid[i].feed_flowrate

        if t == 1:
            prev_flow = (
                m_blk.period[d - 1, m_blk.set_time.last()]
                .reverse_osmosis.ro_skid[i]
                .feed_flowrate
            )
        else:
            prev_flow = m_blk.period[d, t - 1].reverse_osmosis.ro_skid[i].feed_flowrate

        flow_diff = prev_flow - current_flow

        # Capture negative direction of flow change
        return m_blk.flow_changed[d, t, i] * big_M >= flow_diff

    m.num_flow_changes = pyo.Expression(
        expr=1  # Scaling factor (adjust as needed)
        * sum(
            sum(sum(m.flow_changed[d, t, i] for t in m.set_time) for d in m.set_days)
            for i in range(1, m.params.wrd_ro.num_ro_skids + 1)
        )
    )

    m.obj = pyo.Objective(
        expr=m.total_energy_cost
        + m.total_demand_cost
        + m.num_shutdowns
        + m.num_flow_changes,
        sense=pyo.minimize,
    )

    # Can't use gurobi because it requires a liciense for integer variables
    # So going to use ipopt, but may need to look into this further
    # # dt = DiagnosticsToolbox(m)
    # solver = get_solver()
    # results = solver.solve(m)

    mip_gap = 0.03
    solver = pyo.SolverFactory("gurobi_direct_minlp")
    solver.options["MIPGap"] = mip_gap
    results = solver.solve(m, tee=True)

    pyo.assert_optimal_termination(results)

    design_var_values = m.get_design_var_values()
    filtered_design_var_values = {
        k: v
        for k, v in design_var_values.items()
        if "num_flow_changes" not in k and "flow_changed" not in k
    }
    print(filtered_design_var_values)

    plot_function(m, n_time_points=len(price_data))

    # Write optimal values of all operational variables to a csv file
    output_csv = script_dir / "wrd_dummy_result.csv"
    m.get_operation_var_values().to_csv(output_csv)
    print(f"Saved operation variable results to: {output_csv}")

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
    # Return the values of all variables and expressions that do not vary with time

    # OK this runs and solves at least
