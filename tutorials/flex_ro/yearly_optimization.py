# imports
from pyomo.environ import (
    ConcreteModel,
    Param,
    Var,
    Objective,
    Constraint,
    SolverFactory,
    RangeSet,
    NonNegativeReals,
    minimize,
)

from watertap.core.solvers import get_solver

# load surrogate data for summer and winter costs as function of water production (and # of rainy days)


# create surrogate model for both
# For now, I will assign a linear fit for simplicity, but should be rbf (not polynomial!!)
def winter_cost_surrogate(water_production_m3, rainy_days=None):
    # Placeholder linear surrogate model for winter costs
    return 0.320 * water_production_m3 + 2560


def summer_cost_surrogate(water_production_m3, rainy_days=None):
    # Placeholder linear surrogate model for summer costs
    return 0.349 * water_production_m3 + 2346


def apply_water_production_ub(num_rainy_days):
    """Returns an upper bound on water production based on the number of rainy days."""
    # Placeholder linear relationship between rainy days and max water production
    return 376000 - 56080 * num_rainy_days


def init_rainy_days(m, w):
    # This would be replaced with designed rain scenarios or a random distribution
    if w in [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]:
        return 1
    elif w in [25, 26, 27, 28]:
        return 5
    else:
        return 0


def plot_year(m):
    pass


if __name__ == "__main__":
    # Create model and relavant sets/parameters
    m = ConcreteModel()
    m.weeks = RangeSet(1, 52)
    m.week_type = Param(
        m.weeks,
        initialize=lambda m, w: (
            "summer"
            if w in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 49, 50, 51, 52]
            else "winter"
        ),
    )
    m.num_rainy_days = Param(
        m.weeks, initialize=lambda m, w: init_rainy_days(m, w)
    )  # Placeholder

    # Define the variables (water production in each week)
    m.water_production_week = Var(
        m.weeks, bounds=lambda m, w: (0, apply_water_production_ub(m.num_rainy_days[w]))
    )  # m3/week
    m.weekly_cost = Var(m.weeks, bounds=(0, None))  # $/week

    # Add cost as a surrogate models as constraints
    @m.Constraint(m.weeks)
    def eq_cost_per_week(blk, w):
        if m.week_type[w] == "winter":
            return m.weekly_cost[w] == winter_cost_surrogate(m.water_production_week[w])
        else:
            return m.weekly_cost[w] == summer_cost_surrogate(m.water_production_week[w])

    # Add any operational constraints
    # Maximum water deficit?

    # Expressions for total cost and production
    @m.Expression()
    def total_annual_production(blk):
        return sum(m.water_production_week[w] for w in m.weeks)

    @m.Expression()
    def total_cost(blk):
        return sum(m.weekly_cost[w] for w in m.weeks)

    # Add constraint for total annual production
    @m.Constraint()
    def annual_production_target(blk):
        return blk.total_annual_production == 10000 * 1233.5  # Convert AF to m3

    # Define the objective (minimize total cost)
    m.obj = Objective(
        expr=m.total_cost,
        sense=minimize,
    )

    # Solve model w/ ipopt (should work?)
    solver = get_solver()
    results = solver.solve(m, tee=True)

    # Report the results
    # Totals
    print(f"Total annual water production (m3/year): {m.total_annual_production():.2f}")
    print(f"Total annual cost ($/year): {m.total_cost():.2f}")

    # Weekly
    print("Optimal weekly water production (m3/week):")
    for w in m.weeks:
        print(
            f"Week {w}: {m.water_production_week[w]():.2f} m3/week, Cost: ${m.weekly_cost[w]():.2f}, Type: {m.week_type[w]}"
        )

    plot_year(m)
