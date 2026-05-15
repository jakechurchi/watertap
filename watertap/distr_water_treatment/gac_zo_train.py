from pyomo.environ import ConcreteModel, assert_optimal_termination, value
from pyomo.environ import units as pyunits

from idaes.core import FlowsheetBlock, UnitModelCostingBlock
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.core.util.scaling as iscale

from watertap.core import Database, WaterParameterBlock
from watertap.core.solvers import get_solver
from watertap.costing.zero_order_costing import ZeroOrderCosting
from watertap.unit_models.zero_order import GACZO


def build_model(solute_list=None):
    if solute_list is None:
        solute_list = ["tss", "nonvolatile_toc"]

    m = ConcreteModel()
    m.db = Database()

    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.params = WaterParameterBlock(solute_list=solute_list)
    m.fs.unit = GACZO(property_package=m.fs.params, database=m.db)

    return m


def set_operating_conditions(
    m,
    flow_mass_h2o=10000,
    flow_mass_tss=1,
    flow_mass_nonvolatile_toc=1,
    use_default_removal=False,
):
    m.fs.unit.inlet.flow_mass_comp[0, "H2O"].fix(flow_mass_h2o)
    m.fs.unit.inlet.flow_mass_comp[0, "tss"].fix(flow_mass_tss)
    m.fs.unit.inlet.flow_mass_comp[0, "nonvolatile_toc"].fix(flow_mass_nonvolatile_toc)

    m.fs.unit.load_parameters_from_database(use_default_removal=use_default_removal)


def add_costing(m, number_of_parallel_units=5):
    m.fs.costing = ZeroOrderCosting()
    m.fs.unit.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method_arguments={"number_of_parallel_units": number_of_parallel_units},
    )
    m.fs.costing.cost_process()
    m.fs.costing.add_LCOW(m.fs.unit.properties_treated[0].flow_vol)
    m.fs.costing.add_electricity_intensity(m.fs.unit.properties_in[0].flow_vol)


def calculate_scaling_factors(m):
    iscale.calculate_scaling_factors(m)


def initialize(m):
    m.fs.unit.initialize()


def solve(m, solver=None, tee=False):
    if solver is None:
        solver = get_solver()

    results = solver.solve(m, tee=tee)
    assert_optimal_termination(results)
    return results


def report(m):
    print(f"Degrees of freedom: {degrees_of_freedom(m)}")
    m.fs.unit.report()

    t0 = m.fs.time.first()
    print("\nKey results")
    print(
        f"Treated flow volume (m^3/s): "
        f"{value(m.fs.unit.properties_treated[t0].flow_vol):.6f}"
    )
    print(f"Electricity demand (kW): {value(m.fs.unit.electricity[t0]):.3f}")
    print(
        f"Activated carbon demand (kg/h): "
        f"{value(m.fs.unit.activated_carbon_demand[t0]):.6f}"
    )

    if hasattr(m.fs, "costing"):
        print("\nCosting")
        print(
            f"Unit capital cost: "
            f"{value(m.fs.unit.costing.capital_cost):.6f} "
            f"{m.fs.costing.base_currency}"
        )
        if all(
            hasattr(m.fs.unit.costing, c)
            for c in ["contactor_cost", "adsorbent_cost", "other_process_cost"]
        ):
            print(
                f"  C_unit contactor cost: "
                f"{value(m.fs.unit.costing.contactor_cost):.6f} "
                f"{m.fs.costing.base_currency}"
            )
            print(
                f"  C_unit adsorbent cost: "
                f"{value(m.fs.unit.costing.adsorbent_cost):.6f} "
                f"{m.fs.costing.base_currency}"
            )
            print(
                f"  C_unit other process cost: "
                f"{value(m.fs.unit.costing.other_process_cost):.6f} "
                f"{m.fs.costing.base_currency}"
            )
        print(
            f"Total capital cost: "
            f"{value(m.fs.costing.total_capital_cost):.6f} "
            f"{m.fs.costing.base_currency}"
        )
        print(
            f"Total operating cost: "
            f"{value(m.fs.costing.total_operating_cost):.2f} "
            f"{m.fs.costing.base_currency}/{m.fs.costing.base_period}"
        )
        if "activated_carbon" in m.fs.costing.used_flows:
            activated_carbon_cost_per_yr = value(
                pyunits.convert(
                    m.fs.costing.aggregate_flow_costs["activated_carbon"],
                    to_units=pyunits.USD_2018 / pyunits.year,
                )
            )
            print(
                f"Activated carbon cost rate: ${activated_carbon_cost_per_yr:.2f}/year (USD)"
            )
        print(f"LCOW: {value(m.fs.costing.LCOW):.6f} {m.fs.costing.base_currency}/m^3")
        print(
            f"Electricity intensity: "
            f"{value(m.fs.costing.electricity_intensity):.6f} "
            f"kWh/m^3"
        )


if __name__ == "__main__":
    m = build_model()
    set_operating_conditions(
        m,
        flow_mass_h2o=100 * 1000 / 24 / 3600,  # kg/s
        flow_mass_tss=1 / 3600,  # kg/s
        flow_mass_nonvolatile_toc=1 / 3600,  # kg/s
        use_default_removal=False,
    )
    add_costing(m)
    calculate_scaling_factors(m)
    initialize(m)
    solve(m)
    report(m)
