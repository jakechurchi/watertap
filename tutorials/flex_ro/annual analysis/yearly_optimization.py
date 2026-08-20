# imports
import os
import matplotlib.pyplot as plt
from pyomo.environ import (
    ConcreteModel,
    Param,
    Var,
    Objective,
    Constraint,
    RangeSet,
    minimize,
)

from watertap.core.solvers import get_solver

# load surrogate data for summer and winter costs as function of water production (and # of rainy days)
WINTER_INTERP_WATER_PRODUCTION_M3 = [180000, 256000, 330000]
WINTER_INTERP_COST_USD = [200000, 220000, 240000]

SUMMER_INTERP_WATER_PRODUCTION_M3 = [180000, 256000, 330000]
SUMMER_INTERP_COST_USD = [200000, 220000, 240000]


def _line_through_points(x0, y0, x1, y1):
    slope = (y1 - y0) / (x1 - x0)
    intercept = y0 - slope * x0
    return slope, intercept


def _build_segment_lines(production_points, cost_points):
    return [
        _line_through_points(
            production_points[i],
            cost_points[i],
            production_points[i + 1],
            cost_points[i + 1],
        )
        for i in range(len(production_points) - 1)
    ]


WINTER_SEGMENT_LINES = _build_segment_lines(
    WINTER_INTERP_WATER_PRODUCTION_M3, WINTER_INTERP_COST_USD
)
SUMMER_SEGMENT_LINES = _build_segment_lines(
    SUMMER_INTERP_WATER_PRODUCTION_M3, SUMMER_INTERP_COST_USD
)


# create surrogate model for both
# For now, I will assign a linear fit for simplicity, but should be rbf (not polynomial!!)


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


def mid_year_targets(m, weeks, targets_af):
    M3_TO_AF = 1 / 1233.5
    target_map = {w: target / M3_TO_AF for w, target in zip(weeks, targets_af)}

    @m.Constraint(m.weeks)
    def eq_mid_year_targets(blk, w):
        if w in target_map:
            return blk.cumulative_water[w] >= target_map[w]
        return Constraint.Skip


def plot_year(m):
    M3_TO_AF = 1 / 1233.5

    weeks = list(m.weeks)
    cumulative_af = [m.cumulative_water[w]() * M3_TO_AF for w in weeks]
    cumulative_cost = [m.cumulative_cost_var[w]() for w in weeks]

    total_af = m.total_annual_production() * M3_TO_AF
    total_cost = m.total_cost()

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Light grey background for both subplots
    for a in (ax, ax2):
        a.set_facecolor("#f5f5f5")

    # Shade summer weeks on both subplots
    summer_patch = ax.axvspan(
        0.5, 12.5, color="peachpuff", alpha=0.5, label="_nolegend_"
    )
    ax.axvspan(48.5, 52.5, color="peachpuff", alpha=0.5, label="_nolegend_")
    ax2.axvspan(0.5, 12.5, color="peachpuff", alpha=0.5, label="_nolegend_")
    ax2.axvspan(48.5, 52.5, color="peachpuff", alpha=0.5, label="_nolegend_")

    # Shade rainy weeks on top subplot only
    light_blue_patch = None
    dark_blue_patch = None
    for w in weeks:
        rd = m.num_rainy_days[w]
        if rd == 1:
            p = ax.axvspan(
                w - 0.5, w + 0.5, color="lightblue", alpha=0.6, label="_nolegend_"
            )
            if light_blue_patch is None:
                light_blue_patch = p
        elif rd == 5:
            p = ax.axvspan(
                w - 0.5, w + 0.5, color="steelblue", alpha=0.8, label="_nolegend_"
            )
            if dark_blue_patch is None:
                dark_blue_patch = p

    # --- Top subplot: cumulative water production ---
    (line_cum,) = ax.plot(
        weeks, cumulative_af, color="black", linewidth=2, label="Cumulative production"
    )
    line_target = ax.axhline(
        total_af,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Annual target ({total_af:,.0f} AF)",
    )

    # Annotate end-of-quarter cumulative production
    for q_week, q_label in [(13, "End Q1"), (26, "End Q2"), (39, "End Q3")]:
        idx = weeks.index(q_week)
        q_af = cumulative_af[idx]
        ax.annotate(
            f"{q_label}: {q_af:,.0f} AF",
            xy=(q_week, q_af),
            xytext=(q_week + 3, q_af * 0.85),
            fontsize=12,
            arrowprops=dict(arrowstyle="->", color="black"),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"),
        )

    ax.set_ylabel("Water Production (AF)", fontsize=14)
    ax.set_title("Yearly Water Production", fontsize=14)
    ax.tick_params(axis="both", labelsize=14)

    legend_handles = [line_cum, line_target, summer_patch]
    legend_labels = [line_cum.get_label(), line_target.get_label(), "Summer Weeks"]
    if light_blue_patch is not None:
        legend_handles.append(light_blue_patch)
        legend_labels.append("Rainy (1 day)")
    if dark_blue_patch is not None:
        legend_handles.append(dark_blue_patch)
        legend_labels.append("Rainy (5 days)")
    ax.legend(legend_handles, legend_labels, fontsize=14, ncol=2)

    # --- Bottom subplot: cumulative cost + normalized cost ---
    (line_cost,) = ax2.plot(
        weeks,
        [c / 1e6 for c in cumulative_cost],
        color="green",
        linewidth=2,
        label="Cumulative cost",
    )

    ax2.set_xlabel("Week", fontsize=14)
    ax2.set_ylabel("Cumulative Cost (M$)", fontsize=14)
    ax2.set_title("Annual Cost", fontsize=14)
    ax2.set_xlim(0.5, 52.5)
    ax2.set_xticks(range(0, 53, 4))
    ax2.tick_params(axis="both", labelsize=14)

    # Second y-axis: normalized water cost ($/AF)
    ax2b = ax2.twinx()
    norm_cost = [
        (
            m.weekly_cost[w]() / (m.water_production_week[w]() * M3_TO_AF)
            if m.water_production_week[w]() > 0.1
            else float("nan")
        )
        for w in weeks
    ]
    (line_norm,) = ax2b.plot(
        weeks,
        norm_cost,
        color="purple",
        linewidth=2,
        linestyle=":",
        label="Norm. cost ($/AF)",
    )
    ax2b.set_ylabel("Normalized Water Cost ($/AF)", fontsize=14)
    ax2b.tick_params(axis="y", labelsize=14)
    valid = [v for v in norm_cost if v == v]  # filter nan
    if valid:
        ax2b.set_ylim(0, max(valid) * 1.1)

    # Annotate total cost
    ax2.annotate(
        f"Total Cost: ${total_cost:,.0f}",
        xy=(52, total_cost / 1e6),
        xycoords="data",
        xytext=(0.97, 0.28),
        textcoords="axes fraction",
        fontsize=12,
        ha="right",
        arrowprops=dict(arrowstyle="->", color="black"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"),
    )

    handles2, labels2 = ax2.get_legend_handles_labels()
    handles2b, labels2b = ax2b.get_legend_handles_labels()
    ax2b.legend(
        handles2 + handles2b, labels2 + labels2b, fontsize=14, loc="lower right"
    )

    plt.tight_layout()
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "yearly_water_production_cost_plot.png",
    )
    fig.savefig(save_path, dpi=300)
    plt.show()


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

    # Add piecewise-linear cost surrogate constraints.
    # Using linear inequalities plus a minimizing objective keeps the formulation linear.
    @m.Constraint(m.weeks)
    def eq_cost_segment_1(blk, w):
        if m.week_type[w] == "winter":
            slope, intercept = WINTER_SEGMENT_LINES[0]
        else:
            slope, intercept = SUMMER_SEGMENT_LINES[0]
        return m.weekly_cost[w] >= slope * m.water_production_week[w] + intercept

    @m.Constraint(m.weeks)
    def eq_cost_segment_2(blk, w):
        if m.week_type[w] == "winter":
            slope, intercept = WINTER_SEGMENT_LINES[1]
        else:
            slope, intercept = SUMMER_SEGMENT_LINES[1]
        return m.weekly_cost[w] >= slope * m.water_production_week[w] + intercept

    # Add any operational constraints

    # Cumulative water production and cost
    m.cumulative_water = Var(m.weeks, bounds=(0, None))  # m3
    m.cumulative_cost_var = Var(m.weeks, bounds=(0, None))  # $

    @m.Constraint(m.weeks)
    def eq_cumulative_water(blk, w):
        if w == 1:
            return blk.cumulative_water[w] == blk.water_production_week[w]
        return (
            blk.cumulative_water[w]
            == blk.cumulative_water[w - 1] + blk.water_production_week[w]
        )

    @m.Constraint(m.weeks)
    def eq_cumulative_cost(blk, w):
        if w == 1:
            return blk.cumulative_cost_var[w] == blk.weekly_cost[w]
        return (
            blk.cumulative_cost_var[w]
            == blk.cumulative_cost_var[w - 1] + blk.weekly_cost[w]
        )

    mid_year_targets(m, [26], [4000])  # Set mid-year targets in AF

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
