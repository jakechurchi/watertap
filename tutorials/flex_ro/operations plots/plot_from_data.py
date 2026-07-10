## This script is meant to recreate the plotting results from using the csv files.
## The intention is for it to be edited to for customization of the plot formating.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def op_plot_from_data(filename, data_type="optimization_results"):
    script_dir = Path(__file__).resolve().parent
    file_path = Path(filename)
    if not file_path.is_absolute():
        file_path = script_dir / file_path

    df = pd.read_csv(file_path)

    # Column names depend on data_type
    if data_type == "optimization_results":
        column_names = {
            "total_energy": "net_power_consumption",
            "ro1_energy": "reverse_osmosis.ro_skid[1].power_consumption",
            "ro2_energy": "reverse_osmosis.ro_skid[2].power_consumption",
            "ro3_energy": "reverse_osmosis.ro_skid[3].power_consumption",
            "ro4_energy": "reverse_osmosis.ro_skid[4].power_consumption",
            "uf1_energy": "pretreatment.uf_pumps[1].power_consumption",
            "uf2_energy": "pretreatment.uf_pumps[2].power_consumption",
            "uf3_energy": "pretreatment.uf_pumps[3].power_consumption",
            "elec_price": "LMP",
            "prod": "posttreatment.product_flowrate",
            "train_1_flows": "reverse_osmosis.ro_skid[1].product_flowrate",
            "train_2_flows": "reverse_osmosis.ro_skid[2].product_flowrate",
            "train_3_flows": "reverse_osmosis.ro_skid[3].product_flowrate",
            "train_4_flows": "reverse_osmosis.ro_skid[4].product_flowrate",
            "peak_hours": "peak_hour",
        }
    elif data_type == "plant_data":
        # For plant data, flowrates are stored as percentages and power is pre-aggregated
        max_perm_flowrate = 569.0  # m3/hr
        column_names = {
            "total_energy": None,  # Will be computed
            "ro1_energy": "RO_train_1_kW",
            "ro2_energy": "RO_train_2_kW",
            "ro3_energy": "RO_train_3_kW",
            "ro4_energy": "RO_train_4_kW",
            "uf1_energy": "UFFeedPumps_1_kW",
            "uf2_energy": "UFFeedPumps_2_kW",
            "uf3_energy": "UFFeedPumps_3_kW",
            "elec_price": "LMP",
            "prod": None,  # Will be computed from flowrate percentages
            "train_1_flows": ("train_1_flow_pct", max_perm_flowrate),
            "train_2_flows": ("train_2_flow_pct", max_perm_flowrate),
            "train_3_flows": ("train_3_flow_pct", max_perm_flowrate),
            "train_4_flows": ("train_4_flow_pct", max_perm_flowrate),
            "peak_hours": "peak_hour",
        }
        # Compute total energy as sum of all power columns
        power_cols = [
            "RO_train_1_kW",
            "RO_train_2_kW",
            "RO_train_3_kW",
            "RO_train_4_kW",
            "UFFeedPumps_1_kW",
            "UFFeedPumps_2_kW",
            "UFFeedPumps_3_kW",
            "UVAOP_kW",
        ]
        df["_computed_total_energy"] = df[power_cols].sum(axis=1)
        # Compute total flowrate from percentages
        df["_computed_prod"] = (
            df[
                [
                    "train_1_flow_pct",
                    "train_2_flow_pct",
                    "train_3_flow_pct",
                    "train_4_flow_pct",
                ]
            ].sum(axis=1)
            * max_perm_flowrate
            / 100.0
        )
    else:
        raise ValueError(
            f"Unsupported data_type '{data_type}'. Valid options are: "
            "'optimization_results' and 'plant_data'."
        )

    def get_series(series_name, required=True):
        spec = column_names[series_name]

        # Handle computed columns for plant_data
        if spec is None:
            if series_name == "total_energy" and data_type == "plant_data":
                return df["_computed_total_energy"].to_numpy()
            elif series_name == "prod" and data_type == "plant_data":
                return df["_computed_prod"].to_numpy()
            elif required:
                raise KeyError(
                    f"Column for '{series_name}' has not been configured or is not "
                    f"available for data_type='{data_type}'."
                )
            return None

        # Handle percentage-to-flowrate conversion for plant_data
        if isinstance(spec, tuple):
            pct_col, max_flow = spec
            if pct_col not in df.columns:
                raise KeyError(
                    f"Column '{pct_col}' for '{series_name}' was not found in {file_path}."
                )
            return df[pct_col].to_numpy() * max_flow / 100.0

        # Handle regular column lookups
        if spec not in df.columns:
            if required:
                raise KeyError(
                    f"Column '{spec}' for '{series_name}' was not found in {file_path}."
                )
            return None
        return df[spec].to_numpy()

    total_energy = get_series("total_energy")
    n_time_points = len(total_energy)
    output_name = file_path.stem
    output_stem = (
        output_name.split("wrd_result_", 1)[1]
        if "wrd_result_" in output_name
        else output_name
    )

    peak_hours_data = get_series("peak_hours", required=False)
    peak_hours = (
        None
        if peak_hours_data is None or np.all(peak_hours_data == 0)
        else peak_hours_data.astype(bool)
    )
    if peak_hours is None:
        print("Peak hours are missing or all zero; skipping peak-hour shading.")

    time = np.linspace(0, n_time_points - 1, n_time_points)
    fig = plt.figure(figsize=(12, 12))
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
                    label=span_label,
                )
                ax_trains.axvspan(
                    i,
                    i + 1,
                    color="grey",
                    alpha=0.2,
                    linewidth=0,
                    zorder=-1,
                    label=span_label,
                )
                peak_legend_added = True

    # First subplot: Stacked energy consumption by major equipment
    ro1_energy = get_series("ro1_energy")
    ro2_energy = get_series("ro2_energy")
    ro3_energy = get_series("ro3_energy")
    ro4_energy = get_series("ro4_energy")

    uf1_energy = get_series("uf1_energy")
    uf2_energy = get_series("uf2_energy")
    uf3_energy = get_series("uf3_energy")

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
        label="Total Energy Consumption",
        color="black",
        linestyle="--",
        linewidth=2,
    )
    ax_energy.set_ylim(0, 2500)
    ax_energy.set_ylabel("Energy Consumption (kWh)", fontsize=12)
    ax_energy.set_title(
        "Energy Consumption and Electricity Price", fontsize=14, fontweight="bold"
    )
    ax_energy.grid(False)

    ax_price = ax_energy.twinx()
    elec_price = get_series("elec_price", required=False)
    if elec_price is not None and not np.all(np.isnan(elec_price)):
        ax_price.plot(
            time + 0.5,
            elec_price,
            label="Electricity Cost ($/kWh)",
            color="orange",
            linestyle="--",
            linewidth=2,
        )
        ax_price.set_ylabel("Electricity Cost ($/kWh)", fontsize=12)
        ax_price.set_ylim(0, np.nanmax(elec_price) + 0.03)
    else:
        ax_price.set_ylabel("Electricity Cost ($/kWh)", fontsize=12)
        ax_price.set_ylim(0, 1)
        ax_price.set_visible(False)

    handle1, label1 = ax_energy.get_legend_handles_labels()
    handle2, label2 = ax_price.get_legend_handles_labels()
    handles = handle2 + handle1
    labels = label2 + label1
    leg1 = ax_price.legend(
        handles, labels, loc="lower left", framealpha=1.0, ncol=2, fontsize=10
    )
    leg1.set_zorder(1000)
    leg1.get_frame().set_facecolor("white")
    ax_energy.xaxis.set_major_locator(plt.MaxNLocator(24))

    # Second subplot: Water production and RO train flow rates
    prod = get_series("prod")
    ax_trains.plot(
        time + 0.5,
        prod,
        label="Water Production (m$^3$/h)",
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.75,
    )
    ax_trains.set_ylim(0, 2500)
    ax_trains.axhline(
        y=602 * 4,
        color="blue",
        linestyle="--",
        linewidth=2,
        alpha=0.75,
        label="Nominal Plant Capacity (m$^3$/h)",
        zorder=0,
    )
    ax_trains.set_ylabel("Water Production (m$^3$/h)", fontsize=12)
    ax_trains.set_xlabel("Hours", fontsize=12)
    ax_trains.set_title(
        "Water Production & RO Train Flow Rates", fontsize=14, fontweight="bold"
    )
    ax_trains.xaxis.set_major_locator(plt.MaxNLocator(24))
    ax_trains.grid(False)

    # Extract RO train flow rates (m3/hr) for stacked plotting
    train_1_flows = get_series("train_1_flows")
    train_2_flows = get_series("train_2_flows")
    train_3_flows = get_series("train_3_flows")
    train_4_flows = get_series("train_4_flows")

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
        fontsize=11,
        framealpha=1.0,
        ncol=2,
    )
    leg3.get_frame().set_facecolor("white")

    # Set consistent x-axis limits and formatting
    for a in (ax_energy, ax_trains):
        a.set_xlim(0, n_time_points)
        a.xaxis.set_major_locator(plt.MaxNLocator(24))

    # Tick labels for all axes
    for a in (ax_energy, ax_price, ax_trains):
        a.tick_params(axis="both", labelsize=11)

    fig.tight_layout()
    fig.savefig("wrd_aug_week_op_data.png", dpi=600)
    plt.show()


# Optimization Results
# filename = "wrd_result_summer_both_4_flexible_trains.csv"

# Plant Data
filename = "hourly_operation_breakdown_week.csv"

op_plot_from_data(filename, data_type="plant_data")
