import numpy as np
import os
import pandas as pd
import logging
import matplotlib.pyplot as plt

# Pyomo imports
from pyomo.environ import (
    Expression,
    ConcreteModel,
    Var,
    Param,
    units as pyunits,
    Objective,
)
from pyomo.util.check_units import assert_units_consistent
import matplotlib.dates as mdates

from idaes.core import FlowsheetBlock
from pyomo.environ import Var, Binary, Constraint, Objective, value
from watertap.core.util.model_diagnostics import *
from idaes.core.util.model_statistics import *

# IDAES imports
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
from idaes.core.solvers.get_solver import get_solver
import idaes.logger as idaeslog
from pyomo.environ import SolverFactory

if hasattr(pyunits, "USD_2021"):
    CURRENCY_UNIT = pyunits.USD_2021
elif hasattr(pyunits, "USD"):
    CURRENCY_UNIT = pyunits.USD
else:
    pyunits.load_definitions_from_strings(["USD = [currency]"])
    CURRENCY_UNIT = pyunits.USD


def build_feed_sal_vals(n):
    # Completely made up for the moment
    n = int(value(n))
    sal = np.zeros(24) * pyunits.g / pyunits.L
    sal[0:16] = 1 * pyunits.g / pyunits.L
    sal[16:21] = 1 * pyunits.g / pyunits.L
    sal[21:24] = 1 * pyunits.g / pyunits.L

    # Repeat for the 7 days
    if n > 24:
        sal = np.tile(sal, int(np.ceil(n / 24)))
    return sal[:n]


def load_renewable_prod_data(n):
    # Load data for PV and wind production for the week
    # For now, this is just made up data, but eventually we can use my real data
    # NORMALIZED
    n = int(value(n))
    PV_prod = np.zeros(24)
    wind_prod = np.zeros(24)

    # Assume PV production is highest during the day and zero at night
    PV_prod[6:18] = 0.75
    # Assume wind production is more random but has some production at night
    wind_prod[0:4] = 0.50
    wind_prod[4:16] = 0.30
    wind_prod[16:24] = 0.70

    # Repeat for the 7 days
    if n > 24:
        reps = int(np.ceil(n / 24))
        PV_prod = np.tile(PV_prod, reps)
        wind_prod = np.tile(wind_prod, reps)

    return PV_prod[:n], wind_prod[:n]


def build_flowsheet(
    m=None,
    sal_value=1,
    PV_prod=1,
    wind_prod=1,
    PV_CAP=1,
    wind_CAP=1,
    battery_CAP=1,
    SEC=0.1,
):

    if m is None:
        m = ConcreteModel()

    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.time_step = Param(
        initialize=1,
        mutable=True,
        units=pyunits.h,
        doc="Duration of each multiperiod time block",
    )
    # This could very well be a varable, like the capacity of the
    m.fs.total_plant_production_capacity = Param(
        initialize=1.5 * (100 / 24),  # m3 per hour
        units=pyunits.m**3 / pyunits.h,
        doc="Total plant production capacity in m3 per hour",
    )

    m.fs.total_water_production = Var(
        initialize=m.fs.total_plant_production_capacity,
        bounds=(0, m.fs.total_plant_production_capacity),
        units=pyunits.m**3 / pyunits.h,
        doc="Water produced in a hour in m3",
    )

    m.fs.sec = Param(
        initialize=SEC,
        units=pyunits.kWh / pyunits.m**3,
        doc="Specific energy consumption in kWh per m3 of water produced",
    )

    # Create binary variables to indicate if train is on or off
    m.fs.plant_on = Var(
        initialize=1,
        domain=Binary,
        doc="Binary variable indicating if the plant is on",
    )

    # Variables for Battery
    m.fs.battery_level = Var(
        initialize=0,
        bounds=(0, None),
        units=pyunits.kWh,
        doc="Battery charge in kWh",
    )

    m.fs.previous_battery_level = Var(
        initialize=0,
        bounds=(0, None),
        units=pyunits.kWh,
        doc="Battery charge in kWh from previous time step",
    )

    m.fs.battery_charge = Var(
        initialize=0,
        bounds=(0, None),
        units=pyunits.kWh,
        doc="Battery charging energy in each time step",
    )

    m.fs.battery_discharge = Var(
        initialize=0,
        bounds=(0, None),
        units=pyunits.kWh,
        doc="Battery discharging energy in each time step",
    )

    m.fs.is_charging = Var(
        initialize=0,
        domain=Binary,
        doc="Binary indicator for charging mode (1=charging, 0=discharging)",
    )

    # Constraint the upper and lower bounds of battery level
    @m.Constraint(doc="Battery level cannot exceed capacity")
    def eq_battery_capacity(b):
        return b.fs.battery_level <= battery_CAP

    @m.Constraint(doc="Battery level cannot drop below 20% of capacity")
    def eq_battery_non_negative(b):
        return b.fs.battery_level >= 0.2 * battery_CAP

    # Update battery level base on charge
    @m.Constraint(doc="Battery level update constraint")
    def eq_battery_level(b):
        return (
            b.fs.battery_level
            == b.fs.previous_battery_level
            + 0.85 * b.fs.battery_charge
            - b.fs.battery_discharge
        )

    m.fs.battery_power = Param(
        initialize=10,
        mutable=True,
        units=pyunits.kW,
        doc="Maximum power for charging or discharging the battery",
    )

    @m.Constraint(
        doc="Charging is bounded by battery power and active only when is_charging is 1"
    )
    def eq_battery_charge_limit(b):
        return (
            b.fs.battery_charge
            <= b.fs.battery_power * b.fs.time_step * b.fs.is_charging
        )

    @m.Constraint(
        doc="Discharging is bounded by battery power and active only when is_charging is 0"
    )
    def eq_battery_discharge_limit(b):
        return b.fs.battery_discharge <= (
            b.fs.battery_power * b.fs.time_step * (1 - b.fs.is_charging)
        )

    # Function to calculate energy consumption per m3 of water treated.
    # THIS WILL BE DETERMINED LATER BASED ON FLOWSHEET OF EACH PROCESS. SEC or power as a function of salinity.
    # def calculate_power(sal):
    #     # For time being, let's assume this can be made linear so that the problem can be solved as MILP.
    #     return m.fs.total_plant_production_capacity * m.fs.sec

    m.fs.power_consumption = Expression(
        expr=m.fs.plant_on * m.fs.total_plant_production_capacity * m.fs.sec,
        doc="Power consumption in kW",
    )

    m.fs.energy_consumption = Expression(
        expr=m.fs.power_consumption * m.fs.time_step,
        doc="Energy consumption in kWh",
    )

    # Constrain Energy consumption below available renewable energy
    # @m.Constraint(
    #     doc="Energy consumption must be less than or equal to available renewable energy plus battery discharge"
    # )
    # def eq_energy_constraint(b):
    #     return (
    #         b.fs.energy_consumption
    #         <= (PV_prod * PV_CAP + wind_prod * wind_CAP) * b.fs.time_step
    #         - b.fs.battery_change
    #     )

    @m.Constraint(doc="Hourly energy balance with battery charging/discharging")
    def eq_hourly_energy_balance(b):
        return b.fs.energy_consumption == (
            (PV_prod * PV_CAP + wind_prod * wind_CAP) * b.fs.time_step
            + b.fs.battery_discharge
            - b.fs.battery_charge
        )

    # Constrain water production
    @m.Constraint(
        doc="Water production must be equal to plant capacity when plant is on"
    )
    def eq_production_constraint(b):
        return (
            b.fs.total_water_production
            == b.fs.plant_on * b.fs.total_plant_production_capacity
        )

    m.fs.acc_production = Var(
        initialize=0,
        bounds=(0, None),
        units=pyunits.m**3,
        doc="Accumulate water produced in m3",
    )

    m.fs.pre_acc_production = Var(
        initialize=0,
        bounds=(0, None),
        units=pyunits.m**3,
        doc="Accumulate water produced in m3 from previous step",
    )

    m.fs.acc_energy = Var(
        initialize=0,
        bounds=(0, None),
        units=pyunits.kWh,
        doc="Accumulate energy consumption in kWh",
    )

    m.fs.pre_acc_energy = Var(
        initialize=0,
        bounds=(0, None),
        units=pyunits.kWh,
        doc="Accumulate energy consumption in kWh from previous step",
    )

    @m.Constraint(doc="Constraint to calculate total energy consumption")
    def eq_acc_energy(b):
        return b.fs.acc_energy == b.fs.pre_acc_energy + b.fs.energy_consumption

    @m.Constraint(doc="Constraint to accumulate water production")
    def eq_acc_water_prod(b):
        return (
            b.fs.acc_production
            == b.fs.pre_acc_production + b.fs.total_water_production * b.fs.time_step
        )

    # This would only be required if we use a generator of like an LCOE
    # m.fs.energy_cost = Var(
    #     initialize=0,
    #     bounds=(0, None),
    #     units=CURRENCY_UNIT,
    #     doc="Electricity cost for each time step",
    # )

    # @m.Constraint(doc="Energy cost")
    # def eq_energy_cost(b):
    #     return (
    #         b.fs.energy_cost
    #         == b.fs.LCOE_PV * b.fs.PV_prod
    #         + b.fs.LCOE_wind * b.fs.wind_prod
    #         * b.fs.time_step
    #     )

    return m


def get_wrd_variable_pairs(t1, t2):
    # Connect the accumulated water produced
    return [
        (t1.fs.acc_production, t2.fs.pre_acc_production),
        (t1.fs.acc_energy, t2.fs.pre_acc_energy),
        (t1.fs.battery_level, t2.fs.previous_battery_level),
    ]


def unfix_dof(m):
    # Train 1 and 2 are always on, so we only vary the fraction of water treated by train 3 and 4
    # m.fs.water_production_ro_train_3.unfix()
    # m.fs.water_production_ro_train_4.unfix()
    # m.fs.train_3_on.unfix()
    # m.fs.train_4_on.unfix()
    return None


def initialize_mp(m):
    print("Initializing multi-period model...")
    # Check if first time step
    max_flow = 1.5 * 100 / 24  # m3/hr
    m.fs.total_water_production.set_value(max_flow)
    m.fs.plant_on.set_value(1)


def create_mp(
    n_days=1,
    n_time_points=24,
    sal_values=None,
    PV_prod=None,
    wind_prod=None,
    daily_production_target=100 * pyunits.m**3 / pyunits.day,
    total_water_production_target=100 * pyunits.m**3 / pyunits.day,
):
    """
    This function creates a multi-period flowsheet object for a week for the plant. This object contains
    a pyomo model with a block for each time instance.

    Args:
        n_time_points: Number of time blocks to create

    Returns:
        Object containing multi-period vagmd batch flowsheet model
    """
    m = ConcreteModel()

    m.fs = FlowsheetBlock(dynamic=False)

    # Create Variables for sizing the wind and PV systems (and battery eventually)
    m.fs.PV_CAP = Var(
        initialize=10,
        bounds=(0, 10),
        units=pyunits.kW,
        doc="Capacity of PV system in kW",
    )

    m.fs.wind_CAP = Var(
        initialize=10,
        bounds=(0, 10),
        units=pyunits.kW,
        doc="Capacity of wind system in kW",
    )

    m.fs.battery_CAP = Var(
        initialize=10,
        bounds=(0, 20),
        units=pyunits.kWh,
        doc="Capacity of battery system in kWh",
    )
    # m.fs.wind_CAP.fix(0)
    m.fs.mp = MultiPeriodModel(
        n_time_points=n_time_points,
        process_model_func=build_flowsheet,
        linking_variable_func=get_wrd_variable_pairs,
        initialization_func=None,
        unfix_dof_func=unfix_dof,
        outlvl=logging.WARNING,
    )

    """
    Specify the initialization conditions of each period
    """

    flowsheet_options = {
        t: {
            "sal_value": sal_values[t],
            "PV_prod": PV_prod[t],
            "wind_prod": wind_prod[t],
            "PV_CAP": m.fs.PV_CAP,
            "wind_CAP": m.fs.wind_CAP,
            "battery_CAP": m.fs.battery_CAP,
            "SEC": 1,  # This could be made a function of salinity in the future
        }
        for t in range(n_time_points)
    }

    # unfix_dof_options = {t: {} for t in range(n_time_points)}

    m.fs.mp.build_multi_period_model(
        model_data_kwargs=flowsheet_options,
        flowsheet_options=flowsheet_options,
        initialization_options=None,
        unfix_dof_options=None,
    )

    # Tight variable bounds to shrink search space while preserving feasibility.
    # pv_cap_ub = value(m.fs.PV_CAP.ub)
    # wind_cap_ub = value(m.fs.wind_CAP.ub)
    # battery_cap_ub = value(m.fs.battery_CAP.ub)
    # max_sal = max(value(s) for s in sal_values)
    # max_pv_prod = max(PV_prod)
    # max_wind_prod = max(wind_prod)
    # max_energy_consumption_step = 50 * max_sal
    # max_generation_step = max_pv_prod * pv_cap_ub + max_wind_prod * wind_cap_ub

    for t in range(n_time_points):
        initialize_mp(m.fs.mp.blocks[t].process)
        unfix_dof(m.fs.mp.blocks[t].process)

        # b = m.fs.mp.blocks[t].process.fs
        # b.battery_level.setub(battery_cap_ub)
        # b.previous_battery_level.setub(battery_cap_ub)
        # b.battery_change.setlb(-max_energy_consumption_step)
        # b.battery_change.setub(max_generation_step)

    m.fs.mp.blocks[0].process.fs.pre_acc_production.fix(0)
    m.fs.mp.blocks[0].process.fs.pre_acc_energy.fix(0)
    m.fs.mp.blocks[0].process.fs.previous_battery_level.fix(0)
    # Add constraint for every 24 hours of water production
    # m.fs.days = range(int(value(n_days)))
    # @m.Constraint(m.fs.days, doc="Daily production target for each 24-hour period")
    # def daily_production_constraint(b, day):
    #     start_idx = day * 24
    #     end_idx = min((day + 1) * 24, n_time_points)
    #     return (
    #         sum([b.fs.mp.blocks[i].process.fs.total_water_production for i in range(start_idx, end_idx)])
    #         >= pyunits.convert(daily_production_target * pyunits.days, to_units=pyunits.m**3)
    #     )

    @m.Constraint(doc="Total production")
    def total_production(b):
        return sum(
            [
                b.fs.mp.blocks[i].process.fs.total_water_production
                * b.fs.mp.blocks[i].process.fs.time_step
                for i in range(n_time_points)
            ]
        ) >= pyunits.convert(total_water_production_target, to_units=pyunits.m**3)

    # Create a cost expression based on capacity of wind and PV
    m.total_cost = Expression(
        expr=(
            1524 * (CURRENCY_UNIT / pyunits.kW * m.fs.PV_CAP)
            + 5000 * (CURRENCY_UNIT / pyunits.kW) * m.fs.wind_CAP
            + 800 * (CURRENCY_UNIT / pyunits.kWh) * m.fs.battery_CAP
        )
    )

    # Set objective
    m.fs.obj = Objective(expr=m.total_cost)

    return m


def plot_function(
    n_time_points,
    pv_energy_gen,
    wind_energy_gen,
    energy_consumption,
    battery_level,
    sec_kwh_per_m3,
    save_path=None,
):
    time = np.linspace(0, n_time_points - 1, n_time_points)
    pv_energy_gen_vals = [1000 * value(e) for e in pv_energy_gen]
    wind_energy_gen_vals = [1000 * value(e) for e in wind_energy_gen]
    energy_cons_vals = [1000 * value(e) for e in energy_consumption]
    battery_vals = [1000 * value(b) for b in battery_level]

    fig, ax_energy = plt.subplots(1, 1, figsize=(12, 6))

    # --- Top subplot: battery level, power consumption, power generated ---
    ax_energy_right = ax_energy.twinx()

    total_generation_vals = [
        pv + wind for pv, wind in zip(pv_energy_gen_vals, wind_energy_gen_vals)
    ]
    gen_stack = ax_energy.stackplot(
        time,
        pv_energy_gen_vals,
        wind_energy_gen_vals,
        colors=["tab:orange", "tab:purple"],
        alpha=0.5,
        labels=["PV energy generated (Wh)", "Wind energy generated (Wh)"],
    )
    total_gen_line = ax_energy.plot(
        time,
        total_generation_vals,
        color="tab:brown",
        linewidth=2,
        label="Total renewable generation (Wh)",
    )
    cons_line = ax_energy.plot(
        time,
        energy_cons_vals,
        color="tab:blue",
        linewidth=2,
        linestyle="-.",
        label="Energy consumption (Wh)",
    )
    bat_line = ax_energy_right.plot(
        time,
        battery_vals,
        color="tab:green",
        linewidth=2,
        label="Battery level (Wh)",
        linestyle=":",
    )

    ax_energy.set_ylabel("Energy (Wh)", color="tab:orange", fontsize=14)
    ax_energy_right.set_ylabel("Battery level (Wh)", color="tab:green", fontsize=14)
    ax_energy.tick_params(axis="y", labelcolor="tab:orange", labelsize=14)
    ax_energy_right.tick_params(axis="y", labelcolor="tab:green", labelsize=14)
    # Use identical y-axis limits on both axes for direct visual comparison.
    shared_top_max = (
        max(
            max(total_generation_vals),
            max(energy_cons_vals),
            max(battery_vals),
            0,
        )
        + 50
    )
    shared_top_min = min(0, min(battery_vals), min(energy_cons_vals)) - 10
    ax_energy.set_ylim(shared_top_min, shared_top_max)
    ax_energy_right.set_ylim(shared_top_min, shared_top_max)
    ax_energy.grid(alpha=0.3)
    ax_energy.set_xlabel("Time step", fontsize=14)
    ax_energy.tick_params(axis="x", labelsize=14)

    lines_top = [
        cons_line[0],
        total_gen_line[0],
        gen_stack[0],
        gen_stack[1],
        bat_line[0],
    ]
    labels_top = [line.get_label() for line in lines_top]
    ax_energy.legend(lines_top, labels_top, loc="upper left", fontsize=14)
    ax_energy.set_title(
        f"Energy and Battery | SEC: {sec_kwh_per_m3:.2f} kWh/m3", fontsize=14
    )

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    plt.show()


def print_unfixed_vars(model):
    print("Unfixed variables contributing to degrees of freedom:")
    for v in model.component_data_objects(ctype=Var, descend_into=True):
        if not v.fixed:
            print(f"  {v.name}")


if __name__ == "__main__":
    n_days = 3
    n_time_points = 24 * n_days
    daily_production_target = 0 * pyunits.m**3 / pyunits.day
    total_water_production_target = (
        100 * pyunits.m**3 / pyunits.day * n_days * pyunits.day
    )

    sal_values = build_feed_sal_vals(n=n_time_points)
    PV_prod, wind_prod = load_renewable_prod_data(n=n_time_points)

    m = create_mp(
        n_days=n_days,
        n_time_points=n_time_points,
        sal_values=sal_values,
        PV_prod=PV_prod,
        wind_prod=wind_prod,
        daily_production_target=daily_production_target,
        total_water_production_target=total_water_production_target,
    )
    assert_units_consistent(m)
    # print_unfixed_vars(m)

    os.environ["PATH"] = (
        r"C:\Users\rchurchi\AppData\Local\anaconda3\pkgs\glpk-4.65-h17947e8_4\Library\bin"
        + os.pathsep
        + os.environ.get("PATH", "")
    )

    # dt = DiagnosticsToolbox(m)

    # solver = SolverFactory("mindtpy")
    # results = solver.solve(
    #     m,
    #     strategy="OA",
    #     mip_solver="glpk",
    #     nlp_solver="ipopt",
    #     tee=True,
    # )

    solver = SolverFactory("glpk")
    # results = solver.solve(m, tee=True)

    # mip_gap = 0.01
    # solver = SolverFactory("gurobi_direct_minlp")
    # solver.options["MIPGap"] = mip_gap  # 2.0 %
    # solver.options["MIPGapAbs"] = (
    #     0.1  # $1,000 (b/c objective function is scaled down by 1e-4)
    # )
    # solver.options["MIPFocus"] = 1
    results = solver.solve(m, tee=True)

    pv_energy_gen = [
        value(m.fs.PV_CAP) * PV_prod[i] * value(m.fs.mp.blocks[i].process.fs.time_step)
        for i in range(n_time_points)
    ]
    wind_energy_gen = [
        value(m.fs.wind_CAP)
        * wind_prod[i]
        * value(m.fs.mp.blocks[i].process.fs.time_step)
        for i in range(n_time_points)
    ]
    energy_consumption = [
        m.fs.mp.blocks[i].process.fs.energy_consumption for i in range(n_time_points)
    ]
    battery_level = [
        m.fs.mp.blocks[i].process.fs.battery_level for i in range(n_time_points)
    ]

    total_water_production = sum(
        value(m.fs.mp.blocks[i].process.fs.total_water_production)
        * value(m.fs.mp.blocks[i].process.fs.time_step)
        for i in range(n_time_points)
    )
    total_energy_consumption = sum(value(e) for e in energy_consumption)
    sec_kwh_per_m3 = total_energy_consumption / total_water_production

    print("Degrees of freedom:", degrees_of_freedom(m))
    print("Total water production (m3):", total_water_production)
    print("PV size (kW):", value(m.fs.PV_CAP))
    print("Battery size (kWh):", value(m.fs.battery_CAP))
    print("Wind size (kW):", value(m.fs.wind_CAP))
    print("Total target water production in m3:", total_water_production_target())
    print("Total Energy System CAPEX:", m.total_cost(), "2021 $")
    print("SEC (kWh/m3):", sec_kwh_per_m3)

    output_figure_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "weekly_optimization_results.png",
    )
    plot_function(
        n_time_points,
        pv_energy_gen,
        wind_energy_gen,
        energy_consumption,
        battery_level,
        sec_kwh_per_m3,
        save_path=output_figure_path,
    )
