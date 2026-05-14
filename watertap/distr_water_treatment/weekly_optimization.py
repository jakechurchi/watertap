import numpy as np
import os
import pandas as pd
import logging
import matplotlib.pyplot as plt

# Pyomo imports
from pyomo.environ import ConcreteModel, Var, Param, units as pyunits, Objective
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
    sal[16:21] = 2 * pyunits.g / pyunits.L
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
    m=None, sal_value=1, PV_prod=1, wind_prod=1, PV_CAP=1, wind_CAP=1, battery_CAP=1
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

    m.fs.battery_change = Var(
        initialize=0,
        bounds=(None, None),
        units=pyunits.kWh,
        doc="Change in battery charge in kWh",
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
        return b.fs.battery_level == b.fs.previous_battery_level + b.fs.battery_change

    # Function to calculate energy consumption per m3 of water treated.
    # THIS WILL BE DETERMINED LATER BASED ON FLOWSHEET OF EACH PROCESS. SEC or power as a function of salinity.
    def calculate_power(sal):
        # For time being, let's assume this can be made linear so that the problem can be solved as MILP.
        return (50 * sal / (pyunits.g / pyunits.L)) * pyunits.kW

    m.fs.energy_consumption = Expression(
        expr=m.fs.plant_on * calculate_power(sal_value) * m.fs.time_step,
        doc="Energy consumption in kWh",
    )

    # Constrain Energy consumption below available renewable energy
    @m.Constraint(
        doc="Energy consumption must be less than or equal to available renewable energy plus battery discharge"
    )
    def eq_energy_constraint(b):
        return (
            b.fs.energy_consumption
            <= (PV_prod * PV_CAP + wind_prod * wind_CAP) * b.fs.time_step
            - b.fs.battery_change
        )

    @m.Constraint(doc="Battery charge update constraint")
    def eq_battery_charge(b):
        return (
            b.fs.battery_change
            == (PV_prod * PV_CAP + wind_prod * wind_CAP) * b.fs.time_step
            - b.fs.energy_consumption
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
        initialize=100,
        bounds=(0, None),
        units=pyunits.kW,
        doc="Capacity of PV system in kW",
    )

    m.fs.wind_CAP = Var(
        initialize=100,
        bounds=(0, None),
        units=pyunits.kW,
        doc="Capacity of wind system in kW",
    )

    m.fs.battery_CAP = Var(
        initialize=100,
        bounds=(0, None),
        units=pyunits.kWh,
        doc="Capacity of battery system in kWh",
    )

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

    for t in range(n_time_points):
        initialize_mp(m.fs.mp.blocks[t].process)
        unfix_dof(m.fs.mp.blocks[t].process)

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
            1000 * (CURRENCY_UNIT / pyunits.kW * m.fs.PV_CAP)
            + 1500 * (CURRENCY_UNIT / pyunits.kW) * m.fs.wind_CAP
            + 500 * (CURRENCY_UNIT / pyunits.kWh) * m.fs.battery_CAP
        )
    )

    # Set objective
    m.fs.obj = Objective(expr=m.total_cost)

    return m


def plot_function(
    n_time_points,
    water_prod,
    energy_gen,
    energy_consumption,
    battery_level,
    salinity,
    save_path=None,
):
    time = np.linspace(0, n_time_points - 1, n_time_points)
    water_prod_vals = [value(w) for w in water_prod]
    energy_gen_vals = [value(e) for e in energy_gen]
    energy_cons_vals = [value(e) for e in energy_consumption]
    battery_vals = [value(b) for b in battery_level]
    sal_vals = [value(s) for s in salinity]

    fig, (ax_energy, ax_water) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- Top subplot: battery level, power consumption, power generated ---
    ax_energy_right = ax_energy.twinx()

    gen_line = ax_energy.plot(
        time,
        energy_gen_vals,
        color="tab:orange",
        linewidth=2,
        linestyle="--",
        label="Power generated (kW)",
    )
    cons_line = ax_energy.plot(
        time,
        energy_cons_vals,
        color="tab:blue",
        linewidth=2,
        linestyle="-.",
        label="Power consumption (kWh)",
    )
    bat_line = ax_energy_right.plot(
        time,
        battery_vals,
        color="tab:green",
        linewidth=2,
        label="Battery level (kWh)",
    )

    ax_energy.set_ylabel("Power (kW / kWh)", color="tab:orange")
    ax_energy_right.set_ylabel("Battery level (kWh)", color="tab:green")
    ax_energy.tick_params(axis="y", labelcolor="tab:orange")
    ax_energy_right.tick_params(axis="y", labelcolor="tab:green")
    ax_energy.grid(alpha=0.3)

    lines_top = gen_line + cons_line + bat_line
    labels_top = [line.get_label() for line in lines_top]
    ax_energy.legend(lines_top, labels_top, loc="upper left")
    ax_energy.set_title("Energy and Battery")

    # --- Bottom subplot: water production and salinity ---
    ax_water_right = ax_water.twinx()

    water_line = ax_water.plot(
        time,
        water_prod_vals,
        color="tab:blue",
        linewidth=2,
        label="Water production (m3/h)",
    )
    sal_line = ax_water_right.plot(
        time,
        sal_vals,
        color="tab:red",
        linewidth=2,
        linestyle=":",
        label="Salinity (g/L)",
    )

    ax_water.set_xlabel("Time step")
    ax_water.set_ylabel("Water production (m3/h)", color="tab:blue")
    ax_water_right.set_ylabel("Salinity (g/L)", color="tab:red")
    ax_water.tick_params(axis="y", labelcolor="tab:blue")
    ax_water_right.tick_params(axis="y", labelcolor="tab:red")
    ax_water.grid(alpha=0.3)

    lines_bot = water_line + sal_line
    labels_bot = [line.get_label() for line in lines_bot]
    ax_water.legend(lines_bot, labels_bot, loc="upper left")
    ax_water.set_title("Water Production and Salinity")

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
    n_days = 7
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

    # os.environ["PATH"] = (
    #     r"C:\Users\rchurchi\AppData\Local\anaconda3\pkgs\glpk-4.65-h17947e8_4\Library\bin"
    #     + os.pathsep
    #     + os.environ.get("PATH", "")
    # )

    # dt = DiagnosticsToolbox(m)

    # solver = SolverFactory("mindtpy")
    # results = solver.solve(
    #     m,
    #     strategy="OA",
    #     mip_solver="glpk",
    #     nlp_solver="ipopt",
    #     tee=True,
    # )

    # solver = SolverFactory("glpk")
    # results = solver.solve(m, tee=True)

    # mip_gap = 0.01
    solver = SolverFactory("gurobi_direct_minlp")
    # solver.options["MIPGap"] = mip_gap  # 2.0 %
    # solver.options["MIPGapAbs"] = (
    #     0.1  # $1,000 (b/c objective function is scaled down by 1e-4)
    # )
    # solver.options["MIPFocus"] = 1
    results = solver.solve(m, tee=True)

    water_prod = [
        m.fs.mp.blocks[i].process.fs.total_water_production
        for i in range(n_time_points)
    ]
    energy_gen = [
        value(m.fs.PV_CAP) * PV_prod[i] + value(m.fs.wind_CAP) * wind_prod[i]
        for i in range(n_time_points)
    ]
    energy_consumption = [
        m.fs.mp.blocks[i].process.fs.energy_consumption for i in range(n_time_points)
    ]
    battery_level = [
        m.fs.mp.blocks[i].process.fs.battery_level for i in range(n_time_points)
    ]
    salinity = [sal_values[i] for i in range(n_time_points)]

    total_water_production = sum(
        value(m.fs.mp.blocks[i].process.fs.total_water_production)
        * value(m.fs.mp.blocks[i].process.fs.time_step)
        for i in range(n_time_points)
    )

    print("Degrees of freedom:", degrees_of_freedom(m))
    print("Total water production (m3):", total_water_production)
    print("PV size (kW):", value(m.fs.PV_CAP))
    print("Battery size (kWh):", value(m.fs.battery_CAP))
    print("Wind size (kW):", value(m.fs.wind_CAP))
    print("Total target water production in m3:", total_water_production_target())
    print("Total Energy System CAPEX:", m.total_cost(), "2021 $")

    output_figure_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "weekly_optimization_results.png",
    )
    plot_function(
        n_time_points,
        water_prod,
        energy_gen,
        energy_consumption,
        battery_level,
        salinity,
        save_path=output_figure_path,
    )
