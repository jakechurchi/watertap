#################################################################################
# WaterTAP Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################

"""
Copied from the flexible desalination flowsheet, but altered ro surrogate.
"""

from idaes.apps.grid_integration import OperationModel
from pyomo.environ import (
    Constraint,
    Expression,
    Expr_if,
    NonNegativeReals,
    Param,
    Var,
    Binary,
    units as pyunits,
)
from watertap.flowsheets.flex_desal import params as um_params
from watertap.flowsheets.flex_desal import unit_models as um
from watertap.flowsheets.flex_desal.wrd_unit_models import (
    wrd_reverse_osmosis_operation_model,
    wrd_uf_operation_model,
)


def add_operational_cost_expressions(blk, params: um_params.FlexDesalParams):
    """
    Adds cost expressions to the flowsheet
    """
    # Water revenue
    blk.water_revenue = Expression(
        expr=(
            params.product_water_price
            * blk.posttreatment.product_flowrate
            * params.timestep_hours
        ),
        doc="Revenue generated from product water",
    )

    # Customer cost
    blk.customer_cost = Param(
        initialize=0,
        mutable=True,
        doc="Fixed customer cost",
    )

    # Demand response revenue
    blk.demand_response_price = Param(
        initialize=0, mutable=True, doc="Demand-response prices"
    )
    blk.baseline_power = Param(
        initialize=100, mutable=True, doc="Baseline power requirement"
    )
    blk.demand_response_revenue = Expression(
        expr=blk.demand_response_price
        * (blk.baseline_power - blk.power_from_grid)
        * params.timestep_hours,
        doc="Revenue generated from demand response",
    )

    # Cost of emissions
    blk.emissions_intensity = Param(
        initialize=0, mutable=True, units=pyunits.kg / pyunits.kWh
    )
    blk.emissions_cost = Expression(
        expr=(
            blk.emissions_intensity
            * blk.power_from_grid
            * params.timestep_hours
            * params.emissions_cost
            / 907.185  # Conversion factor: $/ton to $/kg
        ),
        doc="Cost associated with carbon emissions",
    )

    # Cost of energy
    blk.LMP = Param(
        initialize=0,
        mutable=True,
        doc="Locational marginal price of electricity [$/kWh]",
    )
    blk.energy_cost = Expression(
        expr=blk.LMP * blk.power_from_grid * params.timestep_hours,
        doc="Cost of electricity purchased from the grid",
    )

    # Demand cost parameters
    blk.fixed_demand_rate = Param(
        initialize=0,
        mutable=True,
        doc="Constant demand tariff",
    )
    blk.variable_demand_rate = Param(
        initialize=0,
        mutable=True,
        doc="Variable demand tariff",
    )


def build_desal_flowsheet(blk, params: um_params.FlexDesalParams):
    """
    Builds a flowsheet instance of the entire desalination process

    Parameters
    ----------
    blk : Block
        Pyomo Block instance

    params : object
        Object containing model parameters
    """

    # Build units
    blk.intake = OperationModel(
        model_func=um.intake_operation_model,
        model_args={"params": params.intake},
    )
    blk.bypass_pretreatment_flow = Var(
        within=NonNegativeReals,
        doc="Flowrate bypassed to brine discharge due to pretreatment shutdown",
    )
    # UF has not been added yet
    blk.pretreatment = OperationModel(
        model_func=wrd_uf_operation_model,
        model_args={"params": params.wrd_uf},
    )
    blk.reverse_osmosis = OperationModel(
        model_func=wrd_reverse_osmosis_operation_model,
        model_args={"params": params.wrd_ro},
    )
    blk.posttreatment = OperationModel(
        model_func=um.posttreatment_operation_model,
        model_args={"params": params},
    )
    blk.brine_discharge = OperationModel(
        model_func=um.brine_discharge_operation_model,
        model_args={"params": params},
    )
    # Flowsheet connections
    blk.arc_intake_pretreatment = Constraint(
        expr=blk.intake.product_flowrate
        == blk.pretreatment.feed_flowrate + blk.bypass_pretreatment_flow,
        doc="intake-pretreatment mass balance",
    )
    blk.suppress_pretreatment_bypass = Constraint(
        expr=blk.bypass_pretreatment_flow
        <= (1 - blk.pretreatment.op_mode) * params.intake.nominal_flowrate
    )
    blk.arc_pretreatment_ro = Constraint(
        expr=blk.pretreatment.product_flowrate == blk.reverse_osmosis.feed_flowrate,
        doc="pretreatment-reverse_osmosis mass balance",
    )
    blk.arc_ro_posttreatment = Constraint(
        expr=blk.reverse_osmosis.product_flowrate == blk.posttreatment.feed_flowrate,
        doc="reverse_osmosis-posttreatment mass balance",
    )
    blk.calculate_brine_discharge = Constraint(
        expr=blk.brine_discharge.feed_flowrate
        == (
            blk.intake.reject_flowrate
            + blk.bypass_pretreatment_flow
            + blk.pretreatment.reject_flowrate
            + blk.reverse_osmosis.reject_flowrate
            + blk.reverse_osmosis.leftover_flow
            + blk.posttreatment.reject_flowrate
        ),
        doc="Computes the total inflow to brine discharge",
    )

    blk.num_skids_online = Expression(
        expr=sum(blk.reverse_osmosis.ro_skid[:].op_mode),
        doc="Calculates the number of skids operating at time t",
    )

    blk.net_power_consumption = Expression(
        expr=blk.intake.power_consumption
        + blk.pretreatment.power_consumption
        + blk.reverse_osmosis.power_consumption
        + blk.posttreatment.power_consumption
        + blk.brine_discharge.power_consumption,
        doc="Net power consumed from the grid",
    )

    if params.include_onsite_solar:
        blk.power_generation = OperationModel(
            model_func=um.power_generation_operation_model,
            model_args={"params": params},
        )
        blk.net_power_consumption += -blk.power_generation.power_utilized

    if params.include_battery:
        blk.battery = OperationModel()
        blk.net_power_consumption += blk.battery.power_charge - blk.battery.discharge

    # Power purchased from the grid
    blk.power_from_grid = Var(
        within=NonNegativeReals,
        units=pyunits.kW,
        doc="Total power purchased from the grid",
    )
    blk.overall_power_balance = Constraint(
        expr=blk.power_from_grid == blk.net_power_consumption,
        doc="Computes the total power purchased from the grid",
    )

    # Add cost expressions
    add_operational_cost_expressions(blk, params)


def add_delayed_startup_constraints(m):
    """Adds the delayed startup constraints to the model"""
    params: um_params.FlexDesalParams = m.params

    # "Shutdown" post-treatment unit if RO startup is initiated
    @m.Constraint(m.period.index_set())
    def posttreatment_unit_commitment(blk, d, t):
        indices = [(d, t - i) for i in range(params.wrd_ro.startup_delay) if t - i > 0]
        return (1 - blk.period[d, t].posttreatment.op_mode) == sum(
            blk.period[p].reverse_osmosis.ro_skid[1].startup for p in indices
        )

    # Brine pump must operate if RO startup is initiated
    @m.Constraint(m.period.index_set())
    def brine_pump_unit_commitment(blk, d, t):
        indices = [(d, t - i) for i in range(params.wrd_ro.startup_delay) if t - i > 0]
        return blk.period[d, t].brine_discharge.op_mode == sum(
            blk.period[p].reverse_osmosis.ro_skid[1].startup for p in indices
        )


def add_demand_and_fixed_costs(m):
    """Adds variables and expressions/constraints for demand and fixed costs"""

    params: um_params.FlexDesalParams = m.params
    m.fixed_demand_cost = Var(
        within=NonNegativeReals,
        doc="Total fixed demand charge value for the entire horizon",
    )
    m.variable_demand_cost = Var(
        within=NonNegativeReals,
        doc="Total variable demand charge value for the entire time horizon",
    )
    m.fixed_monthly_cost = Var(
        within=NonNegativeReals,
        doc="Total customer cost for the entire time horizon",
    )

    @m.Constraint(m.period.index_set())
    def calculate_fixed_demand_cost(blk, d, t):
        return (
            blk.fixed_demand_cost
            >= blk.period[d, t].fixed_demand_rate
            * blk.period[d, t].power_from_grid
            * params.num_months
        )

    @m.Constraint(m.period.index_set())
    def calculate_variable_demand_cost(blk, d, t):
        return (
            blk.variable_demand_cost
            >= blk.period[d, t].variable_demand_rate
            * blk.period[d, t].power_from_grid
            * params.num_months
        )

    m.calculate_fixed_monthly_cost = Constraint(
        expr=m.fixed_monthly_cost == params.fixed_monthly_cost * params.num_months
    )


def add_replacement_costs(m):
    """Adds expressions for replacement costs"""
    params: um_params.WRD_ROParams = m.params.wrd_ro
    # This should be moved elsewhere as "degree of flex doesn't have to be tied just to replacement costs"
    m.raw_degree_of_flex = Expression(
        expr=sum(
            m.period[d, t].reverse_osmosis.ro_skid[i].shutdown
            for d in m.set_days
            for t in m.set_time
            for i in range(1, params.num_ro_skids + 1)
        )
        / (2 * m.params.num_days * params.num_ro_skids),
        doc="Uncapped flexibility metric based on shutdown count",
    )

    # m.degree_of_flex = Expression(
    #     expr=Expr_if(  # Never encountered this function. It might break things
    #         IF_=(m.raw_degree_of_flex <= 1),
    #         THEN_=m.raw_degree_of_flex,
    #         ELSE_=1,
    #     ),
    #     doc="Degree of flexibility capped to [0, 1]",
    # )

    m.degree_of_flex = Param(
        initialize=1,
        doc="Degree of flexibility capped to [0, 1]",
    )

    if params.replacement_types:
        for i, replacement_type in enumerate(params.replacement_types):
            # Create a variable for that replacement type
            setattr(
                m,
                f"replacement_cost_{replacement_type}",
                Param(
                    within=NonNegativeReals,
                    initialize=params.replacement_costs[i],
                    doc=f"Replacement cost for {replacement_type}",
                ),
            )
        # I think adding the degree of flexiblity increases solve time significantly, based on ipopt.
        m.total_replacement_cost = Expression(
            expr=(
                sum(
                    getattr(m, f"replacement_cost_{replacement_type}")
                    / (
                        params.replacement_lifetimes[i]
                        * (
                            1
                            - params.replacement_max_flex_penalty[i] * m.degree_of_flex
                        )
                    )
                    * m.params.num_months
                    / 12
                    for i, replacement_type in enumerate(params.replacement_types)
                )
            ),
            doc="Total replacement costs annualized over the time horizon",
        )


def add_flow_costs(m):
    """Adds expressions for feed and brine costs"""

    m.total_feed_cost = Expression(
        expr=sum(m.period[:, :].intake.feed_cost) * m.params.timestep_hours,
        doc="Total cost of feed water over the time horizon ($)",
    )

    m.total_brine_cost = Expression(
        expr=sum(m.period[:, :].brine_discharge.brine_cost) * m.params.timestep_hours,
        doc="Total cost of brine discharge over the time horizon ($)",
    )

    m.total_chemical_cost = Expression(
        expr=m.params.timestep_hours
        * (
            sum(m.period[:, :].intake.chemical_cost)
            + sum(m.period[:, :].posttreatment.chemical_cost)
        ),
        doc="Total cost of chemicals over the time horizon ($)",
    )


def add_flow_changes_penalty_binary(m):
    # Add binary variables to track flowrate changes between consecutive periods
    m.flow_changed = Var(
        m.set_days,
        m.set_time,
        range(1, m.params.wrd_ro.num_ro_skids + 1),
        within=Binary,
        doc="Binary variable: 1 if RO skid flowrate changes from previous period, 0 otherwise",
    )

    # Add binary variables to track UF pump flowrate changes between consecutive periods
    m.uf_flow_changed = Var(
        m.set_days,
        m.set_time,
        range(1, m.params.wrd_uf.num_uf_pumps + 1),
        within=Binary,
        doc="Binary variable: 1 if UF pump flowrate changes from previous period, 0 otherwise",
    )

    # Add constraints to detect flowrate changes
    # We need a big-M value for the constraint (use maximum flowrate as big-M)
    big_M = m.params.wrd_ro.maximum_flowrate * 2

    @m.Constraint(m.set_days, m.set_time, range(1, m.params.wrd_ro.num_ro_skids + 1))
    def track_flow_changes(m_blk, d, t, i):
        # Skip first time period of first day (no previous period to compare)
        if d == 1 and t == 1:
            return Constraint.Skip

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
            return Constraint.Skip

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

    # Track UF pump flowrate changes
    @m.Constraint(m.set_days, m.set_time, range(1, m.params.wrd_uf.num_uf_pumps + 1))
    def track_uf_flow_changes(m_blk, d, t, i):
        # Skip first time period of first day (no previous period to compare)
        if d == 1 and t == 1:
            return Constraint.Skip

        # Get current and previous period flowrates
        current_flow = m_blk.period[d, t].pretreatment.uf_pumps[i].feed_flowrate

        if t == 1:
            # First hour of a day (not first day), compare to last hour of previous day
            prev_flow = (
                m_blk.period[d - 1, m_blk.set_time.last()]
                .pretreatment.uf_pumps[i]
                .feed_flowrate
            )
        else:
            # Compare to previous hour in same day
            prev_flow = m_blk.period[d, t - 1].pretreatment.uf_pumps[i].feed_flowrate

        # If flows are different, uf_flow_changed must be 1
        flow_diff = current_flow - prev_flow
        big_M_uf = m.params.wrd_uf.maximum_flowrate * 2

        return m_blk.uf_flow_changed[d, t, i] * big_M_uf >= flow_diff

    @m.Constraint(m.set_days, m.set_time, range(1, m.params.wrd_uf.num_uf_pumps + 1))
    def track_uf_flow_changes_neg(m_blk, d, t, i):
        # Skip first time period of first day
        if d == 1 and t == 1:
            return Constraint.Skip

        current_flow = m_blk.period[d, t].pretreatment.uf_pumps[i].feed_flowrate

        if t == 1:
            prev_flow = (
                m_blk.period[d - 1, m_blk.set_time.last()]
                .pretreatment.uf_pumps[i]
                .feed_flowrate
            )
        else:
            prev_flow = m_blk.period[d, t - 1].pretreatment.uf_pumps[i].feed_flowrate

        flow_diff = prev_flow - current_flow
        big_M_uf = m.params.wrd_uf.maximum_flowrate * 2

        # Capture negative direction of flow change
        return m_blk.uf_flow_changed[d, t, i] * big_M_uf >= flow_diff

    # The penalty is simply the number of flow changes multiplied by a scaling factor
    m.flow_changes_penalty = Expression(
        expr=5  # Scaling factor (adjust as needed)
        * (
            sum(
                sum(
                    sum(m.flow_changed[d, t, i] for t in m.set_time) for d in m.set_days
                )
                for i in range(1, m.params.wrd_ro.num_ro_skids + 1)
            )
            + sum(
                sum(
                    sum(m.uf_flow_changed[d, t, i] for t in m.set_time)
                    for d in m.set_days
                )
                for i in range(1, m.params.wrd_uf.num_uf_pumps + 1)
            )
        )
    )


def add_flow_changes_penalty_continuous(m):
    # variables to track flowrate changes between consecutive periods
    m.flow_change = Var(
        m.set_days,
        m.set_time,
        range(1, m.params.wrd_ro.num_ro_skids + 1),
        within=NonNegativeReals,
        doc="Variable to track RO skid flowrate changes between consecutive periods",
    )

    @m.Constraint(m.set_days, m.set_time, range(1, m.params.wrd_ro.num_ro_skids + 1))
    def total_ro_flow_changes(m_blk, d, t, i):
        # Skip first time period of first day (no previous period to compare)
        if d == 1 and t == 1:
            return Constraint.Skip

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

        return m.flow_change[d, t, i] == abs(flow_diff)

    m.flow_changes_penalty = Expression(
        expr=1  # Scaling factor (adjust as needed)
        * (
            sum(
                m.flow_change[d, t, i]
                * (1 - m.period[d, t].reverse_osmosis.ro_skid[i].shutdown)
                * (
                    1 - m.period[d, t].reverse_osmosis.ro_skid[i].startup
                )  # Only 1 when both are 0
                for d in m.set_days
                for t in m.set_time
                for i in range(1, m.params.wrd_ro.num_ro_skids + 1)
            )
        )
    )


def add_useful_expressions(m):
    """Defines useful expressions for custom objective functions"""

    m.total_water_revenue = Expression(expr=sum(m.period[:, :].water_revenue))
    m.total_demand_response_revenue = Expression(
        expr=sum(m.period[:, :].demand_response_revenue)
    )
    m.total_emissions_cost = Expression(expr=sum(m.period[:, :].emissions_cost))


def constrain_water_production(m, baseline_production: float = None):
    """Constrains the total water production rate"""

    params: um_params.FlexDesalParams = m.params
    if baseline_production is not None:
        m.curtailment_fraction = Param(
            initialize=params.curtailment_fraction,
            mutable=True,
            units=pyunits.dimensionless,
            doc="Fraction of water production that is curtailed",
        )

        m.baseline_production = Param(
            initialize=baseline_production,
            mutable=True,
            units=pyunits.m**3,
            doc="Baseline water production",
        )

        m.water_production_target = Constraint(
            expr=m.total_water_production
            >= m.baseline_production * (1 - m.curtailment_fraction)
        )

    elif params.annual_production_AF is not None:
        # Convert production rate from acre-ft/year to m^3/year
        annual_production_m3 = params.annual_production_AF * 1233.48
        m.production_target_abs = Param(
            initialize=annual_production_m3 / 365 * params.num_days,
            mutable=True,
            units=pyunits.m**3,
            doc="Absolute water production target",
        )

        m.water_production_target = Constraint(
            expr=m.total_water_production >= m.production_target_abs
        )

    else:
        raise ValueError("Water production targets not specified in params")
