# imports
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from pyomo.environ import (
    ConcreteModel,
    Param,
    Var,
    Objective,
    Constraint,
    RangeSet,
    maximize,
    minimize,
    value,
)

from watertap.core.solvers import get_solver

# load surrogate data for summer and winter costs as function of water production (and # of rainy days)

## 0 Rainy Days ##
WINTER_0_RAINY_WATER_PRODUCTION_M3 = np.array(
    [
        188120,
        211635,
        235150,
        258665,
        282180,
        305695,
        329210,
        352725,
        376240,
        394716,
    ],
    dtype=float,
)
WINTER_0_RAINY_COST_USD = np.array(
    [
        75507,
        85156,
        93848,
        102961,
        112554.9985,
        121345,
        130424,
        140645,
        149839,
        158104,
    ],
    dtype=float,
)

SUMMER_0_RAINY_WATER_PRODUCTION_M3 = np.array(
    [
        188120,
        211635,
        235150,
        258665,
        282180,
        305695,
        329210,
        352809,
        376240,
        394716,
    ],
    dtype=float,
)
SUMMER_0_RAINY_COST_USD = np.array(
    [
        80542,
        90281,
        99136,
        108042,
        117857,
        126981,
        135947,
        146231,
        159413,
        171054,
    ],
    dtype=float,
)


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


def _validate_piecewise_epigraph(
    production_points,
    cost_points,
    segment_lines,
    dataset_name,
    atol=1e-6,
    ignore_first_segment=False,
):
    """Verify that cost >= max(segment lines) reproduces the supplied knot data.

    This formulation is valid only when the piecewise-linear data are convex, so
    the pointwise maximum of all segment lines matches the knot values.
    """

    start_segment = 1 if ignore_first_segment else 0
    validation_lines = segment_lines[start_segment:]

    slopes = [slope for slope, _ in validation_lines]
    slope_breaks_convexity = [
        (i + start_segment, slopes[i], slopes[i + 1])
        for i in range(len(slopes) - 1)
        if slopes[i] > slopes[i + 1] + atol
    ]

    knot_errors = []
    for index, (x_coord, y_coord) in enumerate(zip(production_points, cost_points)):
        if ignore_first_segment and index == 0:
            continue
        epigraph_value = max(
            slope * x_coord + intercept for slope, intercept in validation_lines
        )
        error = epigraph_value - y_coord
        if abs(error) > atol:
            knot_errors.append((index, x_coord, y_coord, epigraph_value, error))

    if slope_breaks_convexity or knot_errors:
        details = []
        if slope_breaks_convexity:
            first_break = slope_breaks_convexity[0]
            details.append(
                "slopes are not nondecreasing "
                f"(segment {first_break[0]}: {first_break[1]:.6f} > "
                f"segment {first_break[0] + 1}: {first_break[2]:.6f})"
            )
        if knot_errors:
            worst_error = max(knot_errors, key=lambda entry: abs(entry[4]))
            details.append(
                "max(segment lines) does not reproduce the knot data "
                f"(knot {worst_error[0]} at x={worst_error[1]:.3f}: "
                f"expected {worst_error[2]:.6f}, got {worst_error[3]:.6f}, "
                f"error {worst_error[4]:.6f})"
            )
        raise ValueError(
            f"{dataset_name} data are not valid for the current inequality-only "
            "piecewise linearization; "
            + "; ".join(details)
            + ". Use a convex dataset or an exact SOS2/binary piecewise formulation."
        )


WINTER_0_RAINY_SEGMENT_LINES = _build_segment_lines(
    WINTER_0_RAINY_WATER_PRODUCTION_M3, WINTER_0_RAINY_COST_USD
)
SUMMER_0_RAINY_SEGMENT_LINES = _build_segment_lines(
    SUMMER_0_RAINY_WATER_PRODUCTION_M3, SUMMER_0_RAINY_COST_USD
)

_validate_piecewise_epigraph(
    SUMMER_0_RAINY_WATER_PRODUCTION_M3,
    SUMMER_0_RAINY_COST_USD,
    SUMMER_0_RAINY_SEGMENT_LINES,
    "SUMMER_0_RAINY",
    ignore_first_segment=True,
)


_validate_piecewise_epigraph(
    WINTER_0_RAINY_WATER_PRODUCTION_M3,
    WINTER_0_RAINY_COST_USD,
    WINTER_0_RAINY_SEGMENT_LINES,
    "WINTER_0_RAINY",
    ignore_first_segment=True,
)


# For now, I will assign a linear fit for to keep the model linear. However, a rbf surrogate could be trained and used instead.


def apply_water_production_ub(num_rainy_days):
    """Returns an upper bound on water production based on the number of rainy days."""
    # Placeholder linear relationship between rainy days and max water production
    # 394716 is absolute maxium level of water production possible in one week
    # 56080 is the reduction in water production for each additional rainy day

    return 394716 - 56388 * num_rainy_days


def init_rainy_days(m, w):
    # This would be replaced with designed rain scenarios or a random distribution
    if value(m.rainy_days_scenario) == "dry":
        rain_weeks = [0]  # Zero months
    elif value(m.rainy_days_scenario) == "normal":
        rain_weeks = [18, 19, 20, 21, 31, 32, 33, 34]  # Two months
    elif value(m.rainy_days_scenario) == "wet":
        rain_weeks = [
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
        ]  # Four months
    elif value(m.rainy_days_scenario) == "very wet":
        rain_weeks = [
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
        ]  # Five months
    else:
        raise ValueError(
            "Invalid rainy_days_scenario. Choose from 'dry', 'wet', 'very wet', or 'normal'."
        )

    if w in rain_weeks:
        return 7
    else:
        return 0


def mid_year_targets(m, weeks, targets_af):
    M3_TO_AF = 1 / 1233.5
    target_map = {w: target / M3_TO_AF for w, target in zip(weeks, targets_af)}

    @m.Constraint(m.weeks)
    def eq_mid_year_targets(blk, w):
        if w in target_map:
            return (
                blk.cumulative_water[w] == target_map[w]
            )  # Depending on use case, this could be an inequality instead
        return Constraint.Skip


MONTH_TO_WEEKS = {
    1: [1, 2, 3, 4],
    2: [5, 6, 7, 8],
    3: [9, 10, 11, 12, 13],
    4: [14, 15, 16, 17],
    5: [18, 19, 20, 21],
    6: [22, 23, 24, 25, 26],
    7: [27, 28, 29, 30],
    8: [31, 32, 33, 34],
    9: [35, 36, 37, 38, 39],
    10: [40, 41, 42, 43],
    11: [44, 45, 46, 47, 48],
    12: [49, 50, 51, 52],
}


WEEK_TO_MONTH = {
    week: month for month, weeks in MONTH_TO_WEEKS.items() for week in weeks
}


def plot_year(m):
    M3_TO_AF = 1 / 1233.5
    M3_WK_TO_MGD = 264.2 / 10**6 / 7
    weeks = list(m.weeks)
    weeks_with_origin = [0] + weeks
    cumulative_af = [0.0] + [m.cumulative_water[w]() * M3_TO_AF for w in weeks]
    cumulative_cost = [0.0] + [m.cumulative_cost_var[w]() for w in weeks]

    total_af = m.total_annual_production() * M3_TO_AF
    total_cost = m.total_cost()

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Light grey background for both subplots
    for a in (ax, ax2):
        a.set_facecolor("#f5f5f5")

    # Shade summer weeks on both subplots
    summer_patch = ax.axvspan(0, 13, color="peachpuff", alpha=0.5, label="_nolegend_")
    ax.axvspan(48, 52, color="peachpuff", alpha=0.5, label="_nolegend_")
    ax2.axvspan(0, 13, color="peachpuff", alpha=0.5, label="_nolegend_")
    ax2.axvspan(48, 52, color="peachpuff", alpha=0.5, label="_nolegend_")

    # Shade rainy weeks on both subplots
    light_blue_patch = None
    dark_blue_patch = None
    for w in weeks:
        rd = m.num_rainy_days[w]
        if rd == 3:
            p = ax.axvspan(w - 1, w, color="lightblue", alpha=0.6, label="_nolegend_")
            ax2.axvspan(w - 1, w, color="lightblue", alpha=0.6, label="_nolegend_")
            if light_blue_patch is None:
                light_blue_patch = p
        elif rd == 7:
            p = ax.axvspan(w - 1, w, color="steelblue", alpha=0.8, label="_nolegend_")
            ax2.axvspan(w - 1, w, color="steelblue", alpha=0.8, label="_nolegend_")
            if dark_blue_patch is None:
                dark_blue_patch = p

    # --- Top subplot: water production and cumulative production ---
    water_production_week = [m.water_production_week[w]() * M3_WK_TO_MGD for w in weeks]
    ax_right = ax.twinx()
    (line_weekly,) = ax.step(
        weeks,
        water_production_week,
        where="pre",
        color="orange",
        linewidth=2.5,
        linestyle=":",
        label="Weekly production (MGD)",
    )
    ax.set_ylabel("Weekly Water Production (MGD)", fontsize=14)
    ax.set_ylim(bottom=0, top=14.9)

    (line_cum,) = ax_right.plot(
        weeks_with_origin,
        cumulative_af,
        color="black",
        linewidth=2,
        label="Flex cumulative production",
    )
    ax_right.set_ylabel("Cumulative Water (AF)", fontsize=14)
    # line_target = ax.axhline(
    #     total_af,
    #     color="red",
    #     linestyle="--",
    #     linewidth=1.5,
    #     label=f"Annual target ({total_af:,.0f} AF)",
    # )

    # Annotate end-of-quarter cumulative production
    for q_week, q_label in [(13, "End Q1"), (26, "End Q2"), (39, "End Q3")]:
        idx = weeks_with_origin.index(q_week)
        q_af = cumulative_af[idx]
        ax_right.annotate(
            f"{q_label}: {q_af:,.0f} AF",
            xy=(q_week, q_af),
            xytext=(q_week + 3, q_af * 0.85),
            fontsize=12,
            zorder=20,
            arrowprops=dict(arrowstyle="->", color="black"),
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="black",
                zorder=20,
            ),
        )

    ax.set_title(
        f"Rain Scenario: {value(m.rainy_days_scenario)} \n Water Production = {value(m.total_annual_production)/1233.5:.0f} AF",
        fontsize=14,
    )
    ax.tick_params(axis="both", labelsize=14)
    ax_right.tick_params(axis="y", labelsize=14)
    ax.set_ylim(bottom=0)
    ax_right.set_ylim(bottom=0)
    legend_handles = [line_weekly, summer_patch]
    legend_labels = [line_weekly.get_label(), "Summer Weeks"]
    right_handles, right_labels = ax_right.get_legend_handles_labels()
    legend_handles.extend(right_handles)
    legend_labels.extend(right_labels)
    if light_blue_patch is not None:
        legend_handles.append(light_blue_patch)
        legend_labels.append("Rainy (3 day)")
    if dark_blue_patch is not None:
        legend_handles.append(dark_blue_patch)
        legend_labels.append("Rain Shutdown")
    legend = ax.legend(
        legend_handles,
        legend_labels,
        fontsize=14,
        ncol=2,
        loc="lower right",
    )
    legend.set_zorder(10)

    # --- Bottom subplot: cumulative cost + normalized cost ---
    ax2b = ax2.twinx()
    (line_cost,) = ax2b.plot(
        weeks_with_origin,
        [c / 1e6 for c in cumulative_cost],
        color="green",
        linewidth=2,
        label="Flex cumulative cost",
    )
    ax2b.set_ylabel("Cumulative Cost (M$)", fontsize=14)

    ax2.set_xlabel("Week", fontsize=14)
    ax2.set_title("Annual Cost", fontsize=14)
    ax2.set_xlim(0.5, 52)
    ax2.set_xticks(range(0, 53, 4))
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis="both", labelsize=14)
    ax2b.set_ylim(bottom=0)
    ax2b.tick_params(axis="y", labelsize=14)

    # Annotate total cost on the cumulative-cost twin axis so the arrow points
    # to the actual total-cost value rather than the left y-axis baseline.
    ax2.annotate(
        f"Total Cost: ${total_cost:,.0f}",
        xy=(52, total_cost / 1e6),
        xycoords=ax2b.transData,
        xytext=(0.97, 0.35),
        textcoords="axes fraction",
        fontsize=12,
        ha="right",
        arrowprops=dict(arrowstyle="->", color="black"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"),
    )

    # Second y-axis: normalized water cost ($/AF)
    norm_cost = [
        (
            m.weekly_cost[w]() / (m.water_production_week[w]() * M3_TO_AF)
            if m.water_production_week[w]() > 0.1
            else float("nan")
        )
        for w in weeks
    ]
    (line_norm,) = ax2.step(
        weeks,
        norm_cost,
        where="pre",
        color="purple",
        linewidth=2,
        linestyle=":",
        label="Norm. cost ($/AF)",
    )
    ax2.set_ylabel("Normalized Water Cost ($/AF)", fontsize=14)
    ax2.tick_params(axis="y", labelsize=14)
    valid = [v for v in norm_cost if v == v]  # filter nan
    if valid:
        ax2.set_ylim(0, max(valid) * 1.1)

    handles2, labels2 = ax2.get_legend_handles_labels()
    handles2b, labels2b = ax2b.get_legend_handles_labels()
    legend2 = ax2.legend(
        handles2 + handles2b,
        labels2 + labels2b,
        fontsize=14,
        loc="lower right",
    )
    legend2.set_zorder(10)

    plt.tight_layout()
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"{value(m.rainy_days_scenario)}_year_water_production.png",
    )
    fig.savefig(save_path, dpi=300)
    plt.show()


def _load_max_production_profile(m):
    """Load the saved max-production profile for the current rain scenario."""
    scenario_param = getattr(m, "rainy_days_scenario", None)
    if scenario_param is None:
        scenario_param = getattr(m, "rain_day_scenario", None)
    if scenario_param is None:
        raise AttributeError(
            "Model must define rainy_days_scenario or rain_day_scenario."
        )

    scenario_name = str(value(scenario_param)).strip().lower()
    if scenario_name not in {"dry", "normal", "wet"}:
        raise ValueError(
            f"Unsupported rain scenario '{value(scenario_param)}'. "
            "Expected one of: dry, normal, wet."
        )

    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "max_production_profile.csv"
    )
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        normalized = [cell.strip().lower().replace(" ", "") for cell in header]

        water_col = normalized.index(f"{scenario_name}-water")
        cost_col = normalized.index(f"{scenario_name}-cost")

        water = []
        cost = []
        for row in reader:
            if len(row) <= max(water_col, cost_col):
                continue
            try:
                water_value = row[water_col].strip().replace("$", "").replace(",", "")
                cost_value = row[cost_col].strip().replace("$", "").replace(",", "")
                water.append(float(water_value))
                cost.append(float(cost_value))
            except ValueError:
                continue

    if not water or not cost:
        raise ValueError(f"No data found for scenario '{scenario_name}' in {csv_path}.")

    return np.asarray(water, dtype=float), np.asarray(cost, dtype=float)


def plot_year_against_max_strat(m):
    M3_TO_AF = 1 / 1233.5
    M3_WK_TO_MGD = 264.2 / 10**6 / 7
    weeks = list(m.weeks)
    weekly_profile_water, weekly_profile_cost = _load_max_production_profile(m)

    weeks_with_origin = [0] + weeks
    cumulative_af = [0.0] + [m.cumulative_water[w]() * M3_TO_AF for w in weeks]
    cumulative_cost = [0.0] + [m.cumulative_cost_var[w]() for w in weeks]
    max_profile_cum_af = [0.0] + list(np.cumsum(weekly_profile_water) * M3_TO_AF)
    max_profile_cum_cost = [0.0] + list(np.cumsum(weekly_profile_cost))

    total_af = m.total_annual_production() * M3_TO_AF
    total_cost = m.total_cost()

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    for a in (ax, ax2):
        a.set_facecolor("#f5f5f5")

    summer_patch = ax.axvspan(0, 13, color="peachpuff", alpha=0.5, label="_nolegend_")
    ax.axvspan(48, 52, color="peachpuff", alpha=0.5, label="_nolegend_")
    ax2.axvspan(0, 13, color="peachpuff", alpha=0.5, label="_nolegend_")
    ax2.axvspan(48, 52, color="peachpuff", alpha=0.5, label="_nolegend_")

    light_blue_patch = None
    dark_blue_patch = None
    for w in weeks:
        rd = m.num_rainy_days[w]
        if rd == 3:
            p = ax.axvspan(w - 1, w, color="lightblue", alpha=0.6, label="_nolegend_")
            ax2.axvspan(w - 1, w, color="lightblue", alpha=0.6, label="_nolegend_")
            if light_blue_patch is None:
                light_blue_patch = p
        elif rd == 7:
            p = ax.axvspan(w - 1, w, color="steelblue", alpha=0.8, label="_nolegend_")
            ax2.axvspan(w - 1, w, color="steelblue", alpha=0.8, label="_nolegend_")
            if dark_blue_patch is None:
                dark_blue_patch = p

    water_production_week = [m.water_production_week[w]() * M3_WK_TO_MGD for w in weeks]
    ax_right = ax.twinx()
    (line_weekly,) = ax.step(
        weeks,
        water_production_week,
        where="pre",
        color="black",
        linewidth=2.5,
        linestyle=":",
        label="Flex weekly production (MGD)",
    )
    ax.set_ylabel("Weekly Water Production (MGD)", fontsize=14)
    ax.set_ylim(bottom=0, top=14.9)

    (line_cum,) = ax_right.plot(
        weeks_with_origin,
        cumulative_af,
        color="black",
        linewidth=2,
        label="Flex cumulative production",
    )
    (line_max_cum,) = ax_right.plot(
        weeks_with_origin,
        max_profile_cum_af,
        color="tab:red",
        linewidth=2,
        linestyle="-",
        label="Max-strategy cumulative production",
    )
    ax_right.set_ylabel("Cumulative Water (AF)", fontsize=14)

    q_week = 26
    idx = weeks_with_origin.index(q_week)
    q_af = cumulative_af[idx]
    ax_right.annotate(
        f"End Q2: {q_af:,.0f} AF",
        xy=(q_week, q_af),
        xytext=(q_week - 9, q_af * 1),
        fontsize=12,
        zorder=20,
        arrowprops=dict(arrowstyle="->", color="black"),
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="black",
            zorder=20,
        ),
    )

    max_q_af = max_profile_cum_af[idx]
    ax_right.annotate(
        f"Max strategy End Q2: {max_q_af:,.0f} AF",
        xy=(q_week, max_q_af),
        xytext=(q_week, max_q_af * 0.7),
        fontsize=12,
        zorder=20,
        arrowprops=dict(arrowstyle="->", color="tab:red"),
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="tab:red",
            zorder=20,
        ),
    )

    ax.set_title(
        f"Rain Scenario: {value(m.rainy_days_scenario)} \n Water Production = {value(m.total_annual_production)/1233.5:.0f} AF",
        fontsize=14,
    )
    ax.tick_params(axis="both", labelsize=14)
    ax_right.tick_params(axis="y", labelsize=14)
    ax.set_ylim(bottom=0)
    ax_right.set_ylim(bottom=0)
    legend_handles = [line_weekly, summer_patch]
    legend_labels = [line_weekly.get_label(), "Summer Weeks"]
    right_handles, right_labels = ax_right.get_legend_handles_labels()
    legend_handles.extend(right_handles)
    legend_labels.extend(right_labels)
    if light_blue_patch is not None:
        legend_handles.append(light_blue_patch)
        legend_labels.append("Rainy (3 day)")
    if dark_blue_patch is not None:
        legend_handles.append(dark_blue_patch)
        legend_labels.append("Rain Shutdown")
    legend = ax.legend(
        legend_handles,
        legend_labels,
        fontsize=14,
        ncol=2,
        loc="lower right",
    )
    legend.set_zorder(10)

    ax2b = ax2.twinx()
    (line_cost,) = ax2b.plot(
        weeks_with_origin,
        [c / 1e6 for c in cumulative_cost],
        color="green",
        linewidth=2,
        label="Flexible cumulative cost",
    )
    (line_max_cost,) = ax2b.plot(
        weeks_with_origin,
        [c / 1e6 for c in max_profile_cum_cost],
        color="tab:blue",
        linewidth=2,
        linestyle="-",
        label="Max-strategy cumulative cost",
    )
    ax2b.set_ylabel("Cumulative Cost (M$)", fontsize=14)

    ax2.set_xlabel("Week", fontsize=14)
    ax2.set_title("Annual Cost", fontsize=14)
    ax2.set_xlim(0.5, 52)
    ax2.set_xticks(range(0, 53, 4))
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis="both", labelsize=14)
    ax2b.set_ylim(bottom=0)
    ax2b.tick_params(axis="y", labelsize=14)

    max_strategy_total_cost = max_profile_cum_cost[-1]
    total_cost_str = (
        f"Max Strat. Total Cost: ${max_strategy_total_cost:,.0f}\n"
        f"Total Cost: ${total_cost:,.0f}"
    )
    ax2.annotate(
        total_cost_str,
        xy=(52, total_cost / 1e6),
        xycoords=ax2b.transData,
        xytext=(0.98, 0.55),
        textcoords="axes fraction",
        fontsize=12,
        ha="right",
        va="center",
        arrowprops=dict(arrowstyle="->", color="black"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"),
    )

    norm_cost = [
        (
            m.weekly_cost[w]() / (m.water_production_week[w]() * M3_TO_AF)
            if m.water_production_week[w]() > 0.1
            else float("nan")
        )
        for w in weeks
    ]
    (line_norm,) = ax2.step(
        weeks,
        norm_cost,
        where="pre",
        color="purple",
        linewidth=2,
        linestyle=":",
        label="Flex norm. cost ($/AF)",
    )
    ax2.set_ylabel("Normalized Water Cost ($/AF)", fontsize=14)
    ax2.tick_params(axis="y", labelsize=14)
    valid = [v for v in norm_cost if v == v]
    if valid:
        ax2.set_ylim(0, max(valid) * 1.1)

    handles2, labels2 = ax2.get_legend_handles_labels()
    handles2b, labels2b = ax2b.get_legend_handles_labels()
    legend2 = ax2.legend(
        handles2 + handles2b,
        labels2 + labels2b,
        fontsize=14,
        loc="lower right",
    )
    legend2.set_zorder(10)

    plt.tight_layout()
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"{value(m.rainy_days_scenario)}_year_against_max_strat.png",
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
    m.rainy_days_scenario = Param(initialize="very wet")
    m.num_rainy_days = Param(
        m.weeks, initialize=lambda m, w: init_rainy_days(m, w)
    )  # Placeholder
    max_production_scenario = False

    # Define the variables (water production in each week)
    m.months = RangeSet(1, 12)
    m.monthly_water_production = Var(
        m.months, bounds=(0, None)
    )  # m3/month, shared across all weeks in a month
    m.water_production_week = Var(
        m.weeks, bounds=lambda m, w: (0, apply_water_production_ub(m.num_rainy_days[w]))
    )  # m3/week
    m.weekly_cost = Var(m.weeks, bounds=(0, None))  # $/week

    @m.Constraint(m.weeks)
    def eq_monthly_water_production(blk, week):
        return (
            blk.water_production_week[week]
            == blk.monthly_water_production[WEEK_TO_MONTH[week]]
        )

    # Add piecewise-linear cost surrogate constraints.
    # This supports different segment counts for summer and winter.
    max_num_segments = max(
        len(WINTER_0_RAINY_SEGMENT_LINES), len(SUMMER_0_RAINY_SEGMENT_LINES)
    )
    m.cost_segments = RangeSet(0, max_num_segments - 1)

    @m.Constraint(m.weeks, m.cost_segments)
    def eq_cost_segments(blk, w, s):
        if m.week_type[w] == "winter":
            lines = WINTER_0_RAINY_SEGMENT_LINES
        else:
            lines = SUMMER_0_RAINY_SEGMENT_LINES

        if s >= len(lines):
            return Constraint.Skip

        slope, intercept = lines[s]
        return m.weekly_cost[w] >= slope * m.water_production_week[w] + intercept

    # Add any operational constraints

    # Cumulative water production and cost, derived directly from weekly values.
    @m.Expression(m.weeks)
    def cumulative_water(blk, w):
        return sum(blk.water_production_week[ww] for ww in range(1, w + 1))

    @m.Expression(m.weeks)
    def cumulative_cost_var(blk, w):
        return sum(blk.weekly_cost[ww] for ww in range(1, w + 1))

    if max_production_scenario:
        mid_year_targets(
            m, [0], [0]
        )  # Enforces one full production in summer months and
    else:
        mid_year_targets(
            m, [48], [8000]
        )  # Enforces one month of shutdown by reaching target one month early

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
        return blk.total_annual_production == 8000 * 1233.5  # Convert AF to m3

    # Define the objective (minimize total cost)
    if max_production_scenario:
        m.production_penalty = Var(bounds=(0, None), initialize=0)

        @m.Constraint()
        def eq_production_penalty(blk):
            return blk.production_penalty == sum(
                w * m.water_production_week[w] for w in m.weeks
            )

        m.obj = Objective(
            expr=m.total_cost + m.production_penalty,
            sense=minimize,
        )
    else:
        m.obj = Objective(
            expr=m.total_cost,
            sense=minimize,
        )

    # Solve model w/ ipopt (should work?)
    solver = get_solver()
    try:
        results = solver.solve(m, tee=True)
        print(results.solver.termination_condition)
    except Exception as e:
        print(f"Solver failed: {e}")
        print("Falling back to maximum-water-production solve.")
        # Remove the annual production target and maximize water output instead.
        if hasattr(m, "annual_production_target"):
            m.annual_production_target.deactivate()
        if hasattr(m, "eq_mid_year_targets"):
            m.eq_mid_year_targets.deactivate()
        if hasattr(m, "obj"):
            m.obj.deactivate()
        m.max_water_production = Objective(
            expr=10000 - m.total_annual_production,
            sense=minimize,
        )
        results = solver.solve(m, tee=True)
        print(results.solver.termination_condition)

    for month, weeks_in_month in MONTH_TO_WEEKS.items():
        for w in weeks_in_month:
            diff = value(m.water_production_week[w] - m.monthly_water_production[month])
            print(month, w, diff)
            if abs(diff) > 1e-6:
                raise RuntimeError(
                    f"Equal-production constraint violated for month {month}, week {w}: {diff}"
                )

    # Check that cumulative water and cost are consistent with weekly values
    cumulative_water_check = 0.0
    for w in m.weeks:
        cumulative_water_check += value(m.water_production_week[w])
        reported_cumulative_water = value(m.cumulative_water[w])
        diff = reported_cumulative_water - cumulative_water_check
        if abs(diff) > 1e-6:
            raise RuntimeError(
                "Cumulative water profile is inconsistent with weekly production "
                f"at week {w}: reported={reported_cumulative_water:.6f}, "
                f"expected={cumulative_water_check:.6f}, diff={diff:.6f}"
            )

    # Report the results
    # Totals
    print(f"Total annual water production (m3/year): {m.total_annual_production():.2f}")
    print(
        f"Total annual water production (AF/year): {m.total_annual_production() / 1233.5:.2f}"
    )
    print(f"Total annual cost ($/year): {m.total_cost():.2f}")

    # Weekly
    print("Optimal weekly water production (m3/week):")
    for w in m.weeks:
        print(
            f"Week {w}: ,{m.water_production_week[w]():.2f}, m3/week Cost: ,${m.weekly_cost[w]():.2f}, Type: {m.week_type[w]}"
        )

    # Plot the results
    plot_year(m)
