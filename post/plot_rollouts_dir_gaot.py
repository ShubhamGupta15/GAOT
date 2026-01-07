# Standard library.
import os
import sys

# Third party.
import h5py
import matplotlib.pyplot as plt
import numpy as np
import re
import zipfile
import xml.etree.ElementTree as ET

# Local imports (we have to register the repository's main directory first).
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_dir)

# ==================================================================================== #
#                                      Parameters                                      #
# ==================================================================================== #

# Path to the inference directory that holds multiple rollout_*.hdf5 files.
job_inference_dir = r"/home/azureuser/localfiles/GAOT/.inference/examples/time_dep/gis_thermal_sanity"
cfm_zone_xlsx = r"/mnt/eph0/gis_data/cfd_zones/cfd_zones_all_models.xlsx"

# Select which plots to create.
make_mse_plot = True
make_hotspot_plot = True
make_error_dist_plot = True
make_error_dist_over_time_plot = False

# ==================================================================================== #
#                                  Helper functions                                    #
# ==================================================================================== #

GEOMETRY_OPTIONS = [
    "module_short",
    "module_long",
    "module_angle",
    "assembly",
    "long_short",
    "short_angle",
]

_CELL_REF_RE = re.compile(r"^([A-Z]+)([0-9]+)$")


def parse_geometry_and_param(rollout_path: str):
    """Extract geometry and parameter string from a rollout filename (GAOT variant)."""
    name = os.path.splitext(os.path.basename(rollout_path))[0]
    if not name.startswith("rollout_"):
        raise ValueError(f"Rollout filename must start with 'rollout_': {name}")
    name_body = name[len("rollout_") :]
    if "_t0=" in name_body:
        name_body = name_body.split("_t0=", 1)[0]
    geometry = None
    par_str = None

    # Match the geometry by known prefixes to handle geometries with underscores.
    for candidate in sorted(GEOMETRY_OPTIONS, key=len, reverse=True):
        prefix = f"{candidate}_"
        if name_body.startswith(prefix):
            geometry = candidate
            par_str = name_body[len(prefix) :]
            break

    # GAOT rollouts store full sample ids; geometry can be a substring.
    if geometry is None:
        for candidate in sorted(GEOMETRY_OPTIONS, key=len, reverse=True):
            if candidate in name_body:
                geometry = candidate
                par_str = name_body
                break

    if geometry is None or not par_str:
        raise ValueError(f"Could not parse geometry/params from: {name}")

    return geometry, par_str, name


def geometry_to_string(geometry: str) -> str:
    """Map geometry identifier to fancy string."""
    assert geometry in GEOMETRY_OPTIONS, f"Invalid geometry '{geometry}'"

    if geometry == "module_short":
        return "Module Short"
    if geometry == "module_long":
        return "Module Long"
    if geometry == "module_angle":
        return "Module Angle"
    if geometry == "assembly":
        return "Assembly"
    if geometry == "long_short":
        return "Long-Short"
    return "Short-Angle"


def _col_to_index(col: str) -> int:
    idx = 0
    for ch in col:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _load_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        data = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(data)
    strings = []
    for node in root.findall(".//{*}si"):
        texts = [t.text or "" for t in node.findall(".//{*}t")]
        strings.append("".join(texts))
    return strings


def _read_workbook_sheet_path(archive: zipfile.ZipFile, sheet_name: str) -> str:
    workbook_xml = ET.fromstring(archive.read("xl/workbook.xml"))
    rel_id = None
    for sheet in workbook_xml.findall(".//{*}sheet"):
        if sheet.attrib.get("name") == sheet_name:
            rel_id = sheet.attrib.get(
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            )
            break
    if rel_id is None:
        raise ValueError(f"Sheet '{sheet_name}' not found in workbook")

    rels_xml = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    target = None
    for rel in rels_xml.findall(".//{*}Relationship"):
        if rel.attrib.get("Id") == rel_id:
            target = rel.attrib.get("Target")
            break
    if target is None:
        raise ValueError(f"No worksheet target for rel id '{rel_id}'")
    if not target.startswith("xl/"):
        target = f"xl/{target}"
    return target


def _parse_cell_value(cell: ET.Element, shared_strings: list[str]) -> object | None:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        text_nodes = cell.findall(".//{*}t")
        text = "".join([t.text or "" for t in text_nodes])
        return text
    value_node = cell.find("{*}v")
    if value_node is None or value_node.text is None:
        return None
    value = value_node.text
    if cell_type == "s":
        try:
            return shared_strings[int(value)]
        except (ValueError, IndexError):
            return None
    if cell_type == "b":
        return value == "1"
    try:
        number = float(value)
        if number.is_integer():
            return int(number)
        return number
    except ValueError:
        return value


def _read_xlsx_sheet(path: str, sheet_name: str) -> list[list[object | None]]:
    with zipfile.ZipFile(path, "r") as archive:
        shared_strings = _load_shared_strings(archive)
        sheet_path = _read_workbook_sheet_path(archive, sheet_name)
        sheet_xml = ET.fromstring(archive.read(sheet_path))

    rows = []
    sheet_data = sheet_xml.find(".//{*}sheetData")
    if sheet_data is None:
        return rows

    for row in sheet_data.findall("{*}row"):
        row_idx_raw = row.attrib.get("r")
        if row_idx_raw is None:
            continue
        row_idx = int(row_idx_raw) - 1
        if row_idx < 0:
            continue
        while len(rows) <= row_idx:
            rows.append([])
        row_values = rows[row_idx]
        for cell in row.findall("{*}c"):
            cell_ref = cell.attrib.get("r")
            if not cell_ref:
                continue
            match = _CELL_REF_RE.match(cell_ref)
            if not match:
                continue
            col_idx = _col_to_index(match.group(1))
            value = _parse_cell_value(cell, shared_strings)
            if len(row_values) <= col_idx:
                row_values.extend([None] * (col_idx + 1 - len(row_values)))
            row_values[col_idx] = value

    return rows


def load_zone_map(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    rows = _read_xlsx_sheet(path, "Sheet1")
    if not rows:
        return {}
    header = [str(item).strip() if item is not None else "" for item in rows[0]]
    geometry_cols = [idx for idx, name in enumerate(header) if name and name != "Name"]
    zone_maps = {header[idx]: {} for idx in geometry_cols}

    for row in rows[1:]:
        if not row or row[0] is None:
            continue
        name = str(row[0]).strip()
        if name.startswith("solid-"):
            name = name[len("solid-") :]
        for idx in geometry_cols:
            if idx >= len(row):
                continue
            value = row[idx]
            if value in (None, "-", ""):
                continue
            try:
                zone = int(value)
            except (TypeError, ValueError):
                continue
            zone_maps[header[idx]].setdefault(name, []).append(zone)
    return zone_maps


def derive_zone_groups(geometry: str, zone_map: dict) -> dict[str, list[int]]:
    if geometry not in zone_map:
        return {"fluid": [], "conductor": [], "hull": []}
    derived = {"fluid": set(), "conductor": set(), "hull": set()}
    for name, ids in zone_map[geometry].items():
        lname = name.lower()
        if "fluid" in lname:
            target = "fluid"
        elif any(token in lname for token in ("conductor", "aluminum-contact", "contact", "electrode", "pin")):
            target = "conductor"
        else:
            target = "hull"
        derived[target].update(ids)
    return {key: sorted(values) for key, values in derived.items()}


def plot_single_rollout(rollout_hdf5: str, output_dir: str):
    geometry, par_str, rollout_name = parse_geometry_and_param(rollout_hdf5)
    geo_string = geometry_to_string(geometry)

    # Prepare output directory and filenames.
    rollout_results_dir = output_dir
    os.makedirs(rollout_results_dir, exist_ok=True)
    hotspot_png = os.path.join(rollout_results_dir, f"hotspots_{geometry}.png")
    mse_png = os.path.join(rollout_results_dir, f"mse_{geometry}.png")
    error_hist_png = os.path.join(rollout_results_dir, f"error_histogram_{geometry}.png")
    error_hist_t_png = os.path.join(rollout_results_dir, f"error_histogram_over_time_{geometry}.png")

    print(f"\nReading rollout from file: '{rollout_hdf5}'")

    # Read the information from the given hdf5-file.
    with h5py.File(rollout_hdf5, "r") as f:
        T_true = f["temperatures"][:]
        if "temperatures_pred" in f.keys():
            T_pred = f["temperatures_pred"][:]
        else:
            T_pred = f["temperatures"][:]
        node_groups = f["node_group"][:] if "node_group" in f else None
        zone_ids = f["zone_id"][:] if "zone_id" in f else None
        num_edges = len(f["edge_src"][:])
    num_timesteps, num_nodes, _ = T_true.shape

    mask_cond_and_bush = None
    if zone_ids is not None:
        zone_map = load_zone_map(cfm_zone_xlsx)
        zones_cfg = derive_zone_groups(geometry, zone_map)
        if any(zones_cfg.values()):
            mask_fluid = np.isin(zone_ids, zones_cfg["fluid"])
            mask_cond = np.isin(zone_ids, zones_cfg["conductor"])
            mask_hull = np.isin(zone_ids, zones_cfg["hull"])
            mask_bush = np.zeros(num_nodes, dtype=bool)
            mask_cond_and_bush = mask_cond

    if mask_cond_and_bush is None:
        if node_groups is None:
            mask_cond_and_bush = np.ones(num_nodes, dtype=bool)
            mask_cond = mask_cond_and_bush
            mask_hull = np.ones(num_nodes, dtype=bool)
            mask_bush = np.zeros(num_nodes, dtype=bool)
            mask_fluid = np.zeros(num_nodes, dtype=bool)
        else:
            # Derive the masks for conductors + bushings and hull.
            mask_cond_and_bush = np.isin(node_groups, [0, 2])
            mask_cond = node_groups == 0
            mask_hull = node_groups == 1
            mask_bush = node_groups == 2
            mask_fluid = node_groups == 3

    # Derive the node numbers for the different domains.
    num_nodes_cond = mask_cond.sum()
    num_nodes_hull = mask_hull.sum()
    num_nodes_bush = mask_bush.sum()
    num_nodes_fluid = mask_fluid.sum()
    num_nodes_non_external = num_nodes_cond + num_nodes_hull + num_nodes_bush + num_nodes_fluid
    if num_nodes_non_external == 0:
        num_nodes_non_external = num_nodes

    # Get number of timesteps/nodes and prepare time vector (assuming 5min steps).
    print(f"Found {num_timesteps} timesteps for {num_nodes} nodes ({num_edges} edges)")
    t = 5 / 60 * np.arange(1, num_timesteps + 1)
    if num_timesteps >= 600:
        print("Assuming 1-min time steps")
        min_per_timestep = 1
        t /= 5  # In this case, we have 1-min steps.
    else:
        print("Assuming 5-min time steps")
        min_per_timestep = 10
    t_end = min_per_timestep / 60 * num_timesteps

    # Prepare arrays for the conductor hotspot-analysis.
    T_max_cond_pred = np.zeros(num_timesteps)
    T_max_cond_true = np.zeros(num_timesteps)
    T_max_cond_error_max = np.zeros(num_timesteps)
    T_max_cond_error_min = np.zeros(num_timesteps)

    # Prepare arrays for the housing hotspot-analysis.
    T_max_hull_pred = np.zeros(num_timesteps)
    T_max_hull_true = np.zeros(num_timesteps)
    T_max_hull_error_max = np.zeros(num_timesteps)
    T_max_hull_error_min = np.zeros(num_timesteps)

    # Prepare arrays for the housing hotspot-analysis.
    T_max_fluid_pred = np.zeros(num_timesteps)
    T_max_fluid_true = np.zeros(num_timesteps)
    T_max_fluid_error_max = np.zeros(num_timesteps)
    T_max_fluid_error_min = np.zeros(num_timesteps)

    def safe_max(arr):
        return np.max(arr) if arr.size else np.nan

    def safe_min(arr):
        return np.min(arr) if arr.size else np.nan

    # Conduct analysis for each timestep.
    for i in range(num_timesteps):
        # Conductor hotspot predition, ground truth.
        cond_vals_pred = T_pred[i][mask_cond_and_bush]
        cond_vals_true = T_true[i][mask_cond_and_bush]
        T_max_cond_pred[i] = safe_max(cond_vals_pred)
        T_max_cond_true[i] = safe_max(cond_vals_true)

        # Largest hotspot-overestimation and underestimation across conductor components.
        T_max_error = cond_vals_pred - cond_vals_true
        T_max_cond_error_max[i] = safe_max(T_max_error)  # Largest T_max overestimation.
        T_max_cond_error_min[i] = safe_min(T_max_error)  # Largest T_max underestimation.

        # Housing hotspot predition, ground truth.
        hull_vals_pred = T_pred[i][mask_hull]
        hull_vals_true = T_true[i][mask_hull]
        T_max_hull_pred[i] = safe_max(hull_vals_pred)
        T_max_hull_true[i] = safe_max(hull_vals_true)

        # Largest hotspot-overestimation and underestimation across housing components.
        T_max_error = hull_vals_pred - hull_vals_true
        T_max_hull_error_max[i] = safe_max(T_max_error)  # Largest T_max overestimation.
        T_max_hull_error_min[i] = safe_min(T_max_error)  # Largest T_max underestimation.

        # Fluid hotspot predition, ground truth.
        fluid_vals_pred = T_pred[i][mask_fluid]
        fluid_vals_true = T_true[i][mask_fluid]
        T_max_fluid_pred[i] = safe_max(fluid_vals_pred)
        T_max_fluid_true[i] = safe_max(fluid_vals_true)

        # Largest hotspot-overestimation and underestimation across fluid components.
        T_max_error = fluid_vals_pred - fluid_vals_true
        T_max_fluid_error_max[i] = safe_max(T_max_error)  # Largest T_max overestimation.
        T_max_fluid_error_min[i] = safe_min(T_max_error)  # Largest T_max underestimation.

    # Compute the error fields for all three relevant groups.
    dT_cond = T_pred[:, mask_cond_and_bush, :] - T_true[:, mask_cond_and_bush, :]
    dT_hull = T_pred[:, mask_hull, :] - T_true[:, mask_hull, :]
    dT_fluid = T_pred[:, mask_fluid, :] - T_true[:, mask_fluid, :]

    # Compute the MSE over time for each domain.
    error_field = T_pred - T_true
    def safe_mse(field, mask, denom):
        if denom == 0:
            return np.full(num_timesteps, np.nan)
        return np.sum(field[:, mask, :] ** 2, axis=1).flatten() / denom

    mse_over_time_cond = safe_mse(error_field, mask_cond, num_nodes_cond)
    mse_over_time_hull = safe_mse(error_field, mask_hull, num_nodes_hull)
    mse_over_time_bush = safe_mse(error_field, mask_bush, num_nodes_bush)
    mse_over_time_fluid = safe_mse(error_field, mask_fluid, num_nodes_fluid)
    mse_over_time_total = np.sum(error_field ** 2, axis=1).flatten() / num_nodes_non_external

    # Get the cumsums of the MSE-histories.
    timestep_counter = np.arange(1, num_timesteps + 1)
    cmse_over_time_cond = mse_over_time_cond.cumsum() / timestep_counter
    cmse_over_time_hull = mse_over_time_hull.cumsum() / timestep_counter
    cmse_over_time_bush = mse_over_time_bush.cumsum() / timestep_counter
    cmse_over_time_fluid = mse_over_time_fluid.cumsum() / timestep_counter
    cmse_over_time_total = mse_over_time_total.cumsum() / timestep_counter

    # The final values serve as a rollout metric.
    cmse_cond_final = cmse_over_time_cond[-1]
    cmse_hull_final = cmse_over_time_hull[-1]
    cmse_bush_final = cmse_over_time_bush[-1]
    cmse_fluid_final = cmse_over_time_fluid[-1]
    cmse_total_final = cmse_over_time_total[-1]

    # ==================================================================================== #
    #                              Plot: Overall-metric-plot                               #
    # ==================================================================================== #
    if make_mse_plot:
        ax1, ax2 = plt.subplots(1, 2, figsize=(9, 5))[1]

        ax1.plot(t, mse_over_time_cond, label="Conductor", color="tomato", lw=1.5, ls="--")
        ax1.plot(t, mse_over_time_hull, label="Hull", color="orange", lw=1.5, ls="--")
        ax1.plot(t, mse_over_time_bush, label="Bushing", color="darkgray", lw=1.5, ls="--")
        ax1.plot(t, mse_over_time_fluid, label="Fluid", color="navy", lw=1.5, ls="--")
        ax1.plot(t, mse_over_time_total, label="Total", color="black", lw=3)

        ax2.plot(
            t,
            cmse_over_time_cond,
            label=fr"Conductor ({cmse_cond_final:.2f})",
            color="tomato",
            lw=1.5,
            ls="--",
        )
        ax2.plot(
            t,
            cmse_over_time_hull,
            label=fr"Hull ({cmse_hull_final:.2f})",
            color="orange",
            lw=1.5,
            ls="--",
        )
        ax2.plot(
            t,
            cmse_over_time_bush,
            label=fr"Bushing ({cmse_bush_final:.2f})",
            color="darkgray",
            lw=1.5,
            ls="--",
        )
        ax2.plot(
            t,
            cmse_over_time_fluid,
            label=fr"Fluid ({cmse_fluid_final:.2f})",
            color="navy",
            lw=1.5,
            ls="--",
        )
        ax2.plot(
            t,
            cmse_over_time_total,
            label=fr"Total ({cmse_total_final:.2f})",
            color="black",
            lw=3,
        )

        ax2.scatter(x=t[-1], y=cmse_cond_final, color="tomato", s=25)
        ax2.scatter(x=t[-1], y=cmse_hull_final, color="orange", s=25)
        ax2.scatter(x=t[-1], y=cmse_bush_final, color="darkgray", s=25)
        ax2.scatter(x=t[-1], y=cmse_fluid_final, color="navy", s=25)
        ax2.scatter(x=t[-1], y=cmse_total_final, color="black", s=50)

        legend_style = {"fancybox": False, "framealpha": 1, "edgecolor": "k", "fontsize": 10}
        ax1.legend(**legend_style, loc="upper left")
        ax2.legend(**legend_style, loc="upper left")
        ax1.set_title("Mean Squared Error (MSE)")
        ax2.set_title("Time-averaged cumulated MSE")
        for ax in (ax1, ax2):
            ax.set_xlabel("$t$ [h]")
        ax1.set_ylabel(r"MSE(t) [$\mathrm{K}^2$]")
        ax2.set_ylabel(r"TACMSE(t) [$\mathrm{K}^2$]")
        plt.tight_layout()
        print(f"\nWriting MSE-plot to: '{mse_png}'")
        plt.savefig(mse_png, bbox_inches="tight", dpi=300)
        plt.show()

    # ==================================================================================== #
    #                                 Plot: Hotspot-curves                                 #
    # ==================================================================================== #
    if make_hotspot_plot:
        axs = plt.subplots(2, 2, figsize=(9, 6))[1]
        ax1, ax2 = axs[0, 0], axs[0, 1]
        ax3, ax4 = axs[1, 0], axs[1, 1]
        ax1.set_title(f"Hot-spot temperatures\n({geo_string})")
        ax1.set_ylabel("$T$ [K]")
        ax2.set_title(f"GNN prediction errors\n({geo_string})")
        ax2.set_ylabel("$T_\\mathrm{pred} - T_\\mathrm{true}$ [K]")
        ax3.set_ylabel("$T$ [K]")
        ax4.set_ylabel("$T_\\mathrm{pred} - T_\\mathrm{true}$ [K]")

        ax1.plot(t, T_max_cond_true, label="CFD (cond)", color="tomato", lw=2, zorder=3)
        ax1.plot(t, T_max_cond_pred, label="GNN (cond)", color="tomato", lw=2, ls="--", zorder=3)
        ax1.plot(t, T_max_hull_true, label="CFD (hull)", color="orange", lw=2, zorder=2)
        ax1.plot(t, T_max_hull_pred, label="GNN (hull)", color="orange", lw=2, ls="--", zorder=2)
        ax3.plot(t, T_max_fluid_true, label="CFD (hull)", color="navy", lw=2, zorder=1)
        ax3.plot(t, T_max_fluid_pred, label="GNN (hull)", color="navy", lw=2, ls="--", zorder=1)

        ax2.plot(
            t,
            T_max_cond_pred - T_max_cond_true,
            label="hotspot (cond)",
            color="tomato",
            lw=2,
            zorder=2,
        )
        ax2.fill_between(
            t,
            T_max_cond_error_max,
            T_max_cond_error_min,
            color="tomato",
            alpha=0.2,
            label="min/max (cond)",
            zorder=2,
        )
        ax2.plot(
            t,
            T_max_hull_pred - T_max_hull_true,
            label="hotspot (hull)",
            color="orange",
            lw=2,
            zorder=1,
        )
        ax2.fill_between(
            t,
            T_max_hull_error_max,
            T_max_hull_error_min,
            color="orange",
            alpha=0.2,
            label="min/max (hull)",
            zorder=1,
        )
        ax4.plot(
            t,
            T_max_fluid_pred - T_max_fluid_true,
            label="hotspot (fluid)",
            color="navy",
            lw=2,
            zorder=0,
        )
        ax4.fill_between(
            t,
            T_max_fluid_error_max,
            T_max_fluid_error_min,
            color="navy",
            alpha=0.2,
            label="min/max (fluid)",
            zorder=0,
        )

        xlim = ax2.get_xlim()
        ax2.plot(xlim, [0, 0], color="lightgray", zorder=-1, lw=1)
        ax2.set_xlim(xlim)
        ax4.plot(xlim, [0, 0], color="lightgray", zorder=-1, lw=1)
        ax4.set_xlim(xlim)

        legend_style = {"fancybox": False, "framealpha": 1, "edgecolor": "k", "fontsize": 8}
        ax1.legend(**legend_style, loc="upper left")
        ax2.legend(**legend_style, loc="lower left")
        ax3.legend(**legend_style, loc="upper left")
        ax4.legend(**legend_style, loc="lower left")
        for ax in (ax3, ax4):
            ax.set_xlabel("$t$ [h]")
        plt.tight_layout()
        print(f"\nWriting hotspot-plot to: '{hotspot_png}'")
        plt.savefig(hotspot_png, bbox_inches="tight", dpi=300)
        plt.show()

    # ==================================================================================== #
    #                           Plot: Prediction Error Histogram                           #
    # ==================================================================================== #
    if make_error_dist_plot:
        num_bins = None
        (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4.5))[1]

        ax0.hist(
            T_pred[-1, mask_cond_and_bush, 0] - T_true[-1, mask_cond_and_bush, 0],
            num_bins,
            color="tomato",
            edgecolor="black",
        )
        ax0.set_xlabel("$T_\\mathrm{pred} - T_\\mathrm{true}$ [K]")
        ax0.set_ylabel("Count [-]")
        ax0.set_title("Errors on conductors at end of rollout")

        ax1.hist(
            T_pred[-1, mask_hull, 0] - T_true[-1, mask_hull, 0],
            num_bins,
            color="orange",
            edgecolor="black",
        )
        ax1.set_xlabel("$T_\\mathrm{pred} - T_\\mathrm{true}$ [K]")
        ax1.set_ylabel("Count [-]")
        ax1.set_title("Errors on hull at end of rollout")

        ax2.hist(
            T_pred[-1, mask_fluid, 0] - T_true[-1, mask_fluid, 0],
            num_bins,
            color="navy",
            edgecolor="black",
        )
        ax2.set_xlabel("$T_\\mathrm{pred} - T_\\mathrm{true}$ [K]")
        ax2.set_ylabel("Count [-]")
        ax2.set_title("Errors on fluid at end of rollout")

        plt.tight_layout()
        print(f"Writing error-histogram-plot to: '{error_hist_png}'")
        plt.savefig(error_hist_png, bbox_inches="tight", dpi=300)
        plt.show()

    # ==================================================================================== #
    #                           Plot: Prediction Error Histogram                           #
    # ==================================================================================== #
    if make_error_dist_over_time_plot:
        num_bins = 50
        mask_out_zero_bins = False

        error_cond_min = np.min(dT_cond)
        error_cond_max = np.max(dT_cond)
        bins_cond = np.linspace(error_cond_min, error_cond_max, num_bins + 1)
        error_hull_min = np.min(dT_hull)
        error_hull_max = np.max(dT_hull)
        bins_hull = np.linspace(error_hull_min, error_hull_max, num_bins + 1)
        error_fluid_min = np.min(dT_fluid)
        error_fluid_max = np.max(dT_fluid)
        bins_fluid = np.linspace(error_fluid_min, error_fluid_max, num_bins + 1)

        histogram_cond_array = np.zeros((num_timesteps, num_bins))
        histogram_hull_array = np.zeros((num_timesteps, num_bins))
        histogram_fluid_array = np.zeros((num_timesteps, num_bins))

        for i in range(num_timesteps):
            hist, _ = np.histogram(dT_cond[i, :], bins=bins_cond)
            histogram_cond_array[i, :] = hist / hist.max()
            hist, _ = np.histogram(dT_hull[i, :], bins=bins_hull)
            histogram_hull_array[i, :] = hist / hist.max()
            hist, _ = np.histogram(dT_fluid[i, :], bins=bins_fluid)
            histogram_fluid_array[i, :] = hist / hist.max()

        masked_histogram_cond_array = np.ma.masked_where(histogram_cond_array == 0, histogram_cond_array)
        masked_histogram_hull_array = np.ma.masked_where(histogram_hull_array == 0, histogram_hull_array)
        masked_histogram_fluid_array = np.ma.masked_where(histogram_fluid_array == 0, histogram_fluid_array)

        (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(13, 4.5))[1]

        img0 = ax0.imshow(
            masked_histogram_cond_array.T,
            aspect="auto",
            origin="lower",
            extent=[0, t_end, error_cond_min, error_cond_max],
            cmap="Blues",
            interpolation="nearest",
        )
        ax0.set_xlabel("$t$ [h]")
        ax0.set_ylabel(r"$T_\\mathrm{pred} - T_\\mathrm{true}$ [K]")
        ax0.set_title("Conductor error histogram over time")

        img1 = ax1.imshow(
            masked_histogram_hull_array.T,
            aspect="auto",
            origin="lower",
            extent=[0, t_end, error_hull_min, error_hull_max],
            cmap="Blues",
            interpolation="nearest",
        )
        ax1.set_xlabel("$t$ [h]")
        ax1.set_ylabel(r"$T_\\mathrm{pred} - T_\\mathrm{true}$ [K]")
        ax1.set_title("Hull error histogram over time")

        img2 = ax2.imshow(
            masked_histogram_fluid_array.T,
            aspect="auto",
            origin="lower",
            extent=[0, t_end, error_fluid_min, error_fluid_max],
            cmap="Blues",
            interpolation="nearest",
        )
        ax2.set_xlabel("$t$ [h]")
        ax2.set_ylabel(r"$T_\\mathrm{pred} - T_\\mathrm{true}$ [K]")
        ax2.set_title("Fluid error histogram over time")

        if mask_out_zero_bins:
            img0.cmap.set_bad("black")
            img1.cmap.set_bad("black")
            img2.cmap.set_bad("black")

        plt.tight_layout()
        print(f"Writing error-time-histogram-plot to: '{error_hist_t_png}'")
        plt.savefig(error_hist_t_png, bbox_inches="tight", dpi=300)
        plt.show()

    plt.close("all")


def infer_job_name(inference_dir: str) -> str:
    """Derive the job name from an inference directory path."""
    parts = os.path.normpath(inference_dir).split(os.sep)
    if "jobs" in parts:
        idx = parts.index("jobs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return os.path.basename(os.path.dirname(os.path.normpath(inference_dir)))


# ==================================================================================== #
#                                       Run all                                        #
# ==================================================================================== #
if __name__ == "__main__":
    if not os.path.isdir(job_inference_dir):
        raise FileNotFoundError(f"Inference dir not found: {job_inference_dir}")

    job_name = infer_job_name(job_inference_dir)
    job_results_dir = os.path.join(main_dir, ".results", "plots", job_name)
    os.makedirs(job_results_dir, exist_ok=True)

    rollout_files = [
        os.path.join(job_inference_dir, f)
        for f in os.listdir(job_inference_dir)
        if f.startswith("rollout_") and f.endswith(".hdf5")
    ]
    rollout_files.sort()

    if not rollout_files:
        print(f"No rollout_*.hdf5 files found in {job_inference_dir}")
        sys.exit(0)

    print(f"Found {len(rollout_files)} rollout file(s) in {job_inference_dir}")
    for rollout_path in rollout_files:
        try:
            rollout_name = os.path.splitext(os.path.basename(rollout_path))[0]
            rollout_results_dir = os.path.join(job_results_dir, rollout_name)
            plot_single_rollout(rollout_path, rollout_results_dir)
        except Exception as exc:  # pragma: no cover
            print(f"Skipping {rollout_path}: {exc}")
