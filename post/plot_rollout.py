# Standard library.
import sys
import os

# Third party.
import matplotlib.pyplot as plt
import numpy as np
import h5py

# Local imports (we have to register the repository's main directory first).
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_dir)

# ==================================================================================== #
#                                      Parameters                                      #
# ==================================================================================== #
job_name = 'baseline_10'

# Choose your geometry and parameters.
# geometry, par_str = "assembly", "I=3150_T=28.7_r=0.04m_seed=50_t0=0_dt=5min"
# geometry, par_str = "module_short", "I=3150_T=28.7_r=0.02m_seed=0_t0=0_dt=5min"

# geometry, par_str = 'module_angle', 'I=3500_T=45_r=0.04m_seed=0_t0=0_dt=10min'
# geometry, par_str = 'module_long', 'I=5000_T=50_r=0.04m_seed=0_t0=0_dt=10min'
# geometry, par_str = 'module_short', 'I=4000_T=50_r=0.04m_seed=0_t0=0_dt=10min'
# geometry, par_str = 'short_angle', 'I=3150_T=35_r=0.04m_seed=0_t0=0_dt=10min'
geometry, par_str = 'long_short', 'I=4000_T=35_r=0.04m_seed=0_t0=0_dt=10min'
# This is the graph HDF5-file to read from.
# rollout_hdf5 = rf"/home/azureuser/localfiles/graph-trainer-alex/jobs/{job_name}/inference/interpolated_dt5min/rollout_{geometry}_{par_str}.hdf5"
# rollout_hdf5 = rf"/home/azureuser/localfiles/graph-trainer-alex/jobs/{job_name}/inference/rollout_{geometry}_{par_str}.hdf5"
rollout_hdf5 = rf"/home/azureuser/cloudfiles/code/Users/shubham.sg.gupta.ext/upsample/debug/rollout_{geometry}_{par_str}.hdf5"

# This is where the plots are written to.
hotspot_png = os.path.join(main_dir, f"results/plots/hotspots_{geometry}.png")
mse_png = os.path.join(main_dir, f"results/plots/mse_{geometry}.png")
error_hist_png = os.path.join(main_dir, f"results/plots/error_histogram_{geometry}.png")
error_hist_t_png = os.path.join(main_dir, f"results/plots/error_histogram_over_time_{geometry}.png")

# Select which plots to create.
make_mse_plot = True
make_hotspot_plot = True
make_error_dist_plot = True
make_error_dist_over_time_plot = False

# ==================================================================================== #
#                                    Derived values                                    #
# ==================================================================================== #

# Check the given geometry name.
assert geometry in [
    "module_short",
    "module_long",
    "module_angle",
    "assembly",
    "long_short",
    "short_angle",
], f"Invalid geometry '{geometry}'"

# Derive a 'fancy' geometry string
if geometry == "module_short":
    geo_string = "Module Short"
elif geometry == "module_long":
    geo_string = "Module Long"
elif geometry == "module_angle":
    geo_string = "Module Angle"
elif geometry == "assembly":
    geo_string = "Assembly"
elif geometry == "long_short":
    geo_string = "Long-Short"
else:
    assert geometry == "short_angle"
    geo_string = "Short-Angle"

# ==================================================================================== #
#                              Read-in / preprocess data                               #
# ==================================================================================== #

# Select the rollout-file either from the Downloads-folder or from the code-dir.
print(f"\nReading rollout from file: '{rollout_hdf5}'")

# Read the information from the given hdf5-file.
with h5py.File(rollout_hdf5, "r") as f:
    T_true = f["temperatures"][:]
    if "temperatures_pred" in f.keys():
        T_pred = f["temperatures_pred"][:]
    else:
        T_pred = f["temperatures"][:]
    node_groups = f["node_group"][:]
    num_edges = len(f["edge_src"][:])
num_timesteps, num_nodes, _ = T_true.shape

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

# ==================================================================================== #
#                              Analysze prediciton errors                              #
# ==================================================================================== #

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

# Conduct analysis for each timestep.
for i in range(num_timesteps):

    # Conductor hotspot predition, ground truth.
    T_max_cond_pred[i] = np.max(T_pred[i][mask_cond_and_bush])
    T_max_cond_true[i] = np.max(T_true[i][mask_cond_and_bush])

    # Largest hotspot-overestimation and underestimation across conductor components.
    T_max_error = T_pred[i][mask_cond_and_bush] - T_true[i][mask_cond_and_bush]
    T_max_cond_error_max[i] = np.max(T_max_error)  # Largest T_max overestimation.
    T_max_cond_error_min[i] = np.min(T_max_error)  # Largest T_max underestimation.

    # Housing hotspot predition, ground truth.
    T_max_hull_pred[i] = np.max(T_pred[i][mask_hull])
    T_max_hull_true[i] = np.max(T_true[i][mask_hull])

    # Largest hotspot-overestimation and underestimation across housing components.
    T_max_error = T_pred[i][mask_hull] - T_true[i][mask_hull]
    T_max_hull_error_max[i] = np.max(T_max_error)  # Largest T_max overestimation.
    T_max_hull_error_min[i] = np.min(T_max_error)  # Largest T_max underestimation.

    # Housing hotspot predition, ground truth.
    T_max_fluid_pred[i] = np.max(T_pred[i][mask_fluid])
    T_max_fluid_true[i] = np.max(T_true[i][mask_fluid])

    # Largest hotspot-overestimation and underestimation across housing components.
    T_max_error = T_pred[i][mask_fluid] - T_true[i][mask_fluid]
    T_max_fluid_error_max[i] = np.max(T_max_error)  # Largest T_max overestimation.
    T_max_fluid_error_min[i] = np.min(T_max_error)  # Largest T_max underestimation.

# Compute the error fields for all three relevant groups.
dT_cond = T_pred[:, mask_cond_and_bush, :] - T_true[:, mask_cond_and_bush, :]
dT_hull = T_pred[:, mask_hull, :] - T_true[:, mask_hull, :]
dT_fluid = T_pred[:, mask_fluid, :] - T_true[:, mask_fluid, :]

# Compute the MSE over time for each domain.
error_field = T_pred - T_true
mse_over_time_cond = np.sum(error_field[:, mask_cond, :] ** 2, axis=1).flatten() / num_nodes_cond
mse_over_time_hull = np.sum(error_field[:, mask_hull, :] ** 2, axis=1).flatten() / num_nodes_hull
mse_over_time_bush = np.sum(error_field[:, mask_bush, :] ** 2, axis=1).flatten() / num_nodes_bush
mse_over_time_fluid = np.sum(error_field[:, mask_fluid, :] ** 2, axis=1).flatten() / num_nodes_fluid
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

# Only plot if selected.
if make_mse_plot:

    # Prepare the plot.
    ax1, ax2 = plt.subplots(1, 2, figsize=(9, 5))[1]

    ax1.plot(t, mse_over_time_cond, label="Conductor", color="tomato", lw=1.5, ls="--")
    ax1.plot(t, mse_over_time_hull, label="Hull", color="orange", lw=1.5, ls="--")
    ax1.plot(t, mse_over_time_bush, label="Bushing", color="darkgray", lw=1.5, ls="--")
    ax1.plot(t, mse_over_time_fluid, label="Fluid", color="navy", lw=1.5, ls="--")
    ax1.plot(t, mse_over_time_total, label="Total", color="black", lw=3)

    ax2.plot(t, cmse_over_time_cond, label=fr"Conductor ({cmse_cond_final:.2f})", color="tomato", lw=1.5, ls="--")
    ax2.plot(t, cmse_over_time_hull, label=fr"Hull ({cmse_hull_final:.2f})", color="orange", lw=1.5, ls="--")
    ax2.plot(t, cmse_over_time_bush, label=fr"Bushing ({cmse_bush_final:.2f})", color="darkgray", lw=1.5, ls="--")
    ax2.plot(t, cmse_over_time_fluid, label=fr"Fluid ({cmse_fluid_final:.2f})", color="navy", lw=1.5, ls="--")
    ax2.plot(t, cmse_over_time_total, label=fr"Total ({cmse_total_final:.2f})", color="black", lw=3)

    ax2.scatter(x=t[-1], y=cmse_cond_final, color="tomato", s=25)
    ax2.scatter(x=t[-1], y=cmse_hull_final, color="orange", s=25)
    ax2.scatter(x=t[-1], y=cmse_bush_final, color="darkgray", s=25)
    ax2.scatter(x=t[-1], y=cmse_fluid_final, color="navy", s=25)
    ax2.scatter(x=t[-1], y=cmse_total_final, color="black", s=50)

    # Finalize and show the plot.
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

# Only plot if selected.
if make_hotspot_plot:

    # Prepare the plot with two subplots.
    axs = plt.subplots(2, 2, figsize=(9, 6))[1]
    ax1, ax2 = axs[0, 0], axs[0, 1]
    ax3, ax4 = axs[1, 0], axs[1, 1]
    ax1.set_title(f"Hot-spot temperatures\n({geo_string})")
    ax1.set_ylabel("$T$ [K]")
    ax2.set_title(f"GNN prediction errors\n({geo_string})")
    ax2.set_ylabel("$T_\mathrm{pred} - T_\mathrm{true}$ [K]")
    ax3.set_ylabel("$T$ [K]")
    ax4.set_ylabel("$T_\mathrm{pred} - T_\mathrm{true}$ [K]")

    # Plot the hotspot temperatures (left).
    ax1.plot(t, T_max_cond_true, label="CFD (cond)", color="tomato", lw=2, zorder=3)
    ax1.plot(t, T_max_cond_pred, label="GNN (cond)", color="tomato", lw=2, ls="--", zorder=3)
    ax1.plot(t, T_max_hull_true, label="CFD (hull)", color="orange", lw=2, zorder=2)
    ax1.plot(t, T_max_hull_pred, label="GNN (hull)", color="orange", lw=2, ls="--", zorder=2)
    ax3.plot(t, T_max_fluid_true, label="CFD (hull)", color="navy", lw=2, zorder=1)
    ax3.plot(t, T_max_fluid_pred, label="GNN (hull)", color="navy", lw=2, ls="--", zorder=1)

    # Plot the error curves (right).
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

    # Plot the zero-line in the error plot.
    xlim = ax2.get_xlim()
    ax2.plot(xlim, [0, 0], color="lightgray", zorder=-1, lw=1)
    ax2.set_xlim(xlim)
    ax4.plot(xlim, [0, 0], color="lightgray", zorder=-1, lw=1)
    ax4.set_xlim(xlim)

    # Finalize and show the plot.
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

# Only plot if selected.
if make_error_dist_plot:

    # Change if necessary.
    num_bins = None

    # Prepare the plot.
    (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4.5))[1]

    # Create histogram for conductor nodes.
    ax0.hist(
        T_pred[-1, mask_cond_and_bush, 0] - T_true[-1, mask_cond_and_bush, 0],
        num_bins,
        color="tomato",
        edgecolor="black",
    )
    ax0.set_xlabel("$T_\mathrm{pred} - T_\mathrm{true}$ [K]")
    ax0.set_ylabel("Count [-]")
    ax0.set_title(f"Errors on conductors at end of rollout")

    # Create histogram for hull nodes.
    ax1.hist(
        T_pred[-1, mask_hull, 0] - T_true[-1, mask_hull, 0],
        num_bins,
        color="orange",
        edgecolor="black",
    )
    ax1.set_xlabel("$T_\mathrm{pred} - T_\mathrm{true}$ [K]")
    ax1.set_ylabel("Count [-]")
    ax1.set_title(f"Errors on hull at end of rollout")

    # Create histogram for fluid nodes.
    ax2.hist(
        T_pred[-1, mask_fluid, 0] - T_true[-1, mask_fluid, 0],
        num_bins,
        color="navy",
        edgecolor="black",
    )
    ax2.set_xlabel("$T_\mathrm{pred} - T_\mathrm{true}$ [K]")
    ax2.set_ylabel("Count [-]")
    ax2.set_title(f"Errors on fluid at end of rollout")

    # Finalize and show/write plot.
    plt.tight_layout()
    print(f"Writing error-histogram-plot to: '{error_hist_png}'")
    plt.savefig(error_hist_png, bbox_inches="tight", dpi=300)
    plt.show()

# ==================================================================================== #
#                           Plot: Prediction Error Histogram                           #
# ==================================================================================== #

# Only plot if selected.
if make_error_dist_over_time_plot:

    # Change if necessary.
    num_bins = 50
    mask_out_zero_bins = False

    # Define the bins for the histograms.
    error_cond_min = np.min(dT_cond)
    error_cond_max = np.max(dT_cond)
    bins_cond = np.linspace(error_cond_min, error_cond_max, num_bins + 1)
    error_hull_min = np.min(dT_hull)
    error_hull_max = np.max(dT_hull)
    bins_hull = np.linspace(error_hull_min, error_hull_max, num_bins + 1)
    error_fluid_min = np.min(dT_fluid)
    error_fluid_max = np.max(dT_fluid)
    bins_fluid = np.linspace(error_fluid_min, error_fluid_max, num_bins + 1)

    # Initialize the 2D array to store histograms for each time step.
    histogram_cond_array = np.zeros((num_timesteps, num_bins))
    histogram_hull_array = np.zeros((num_timesteps, num_bins))
    histogram_fluid_array = np.zeros((num_timesteps, num_bins))

    # Compute the histogram for each time step.
    for i in range(num_timesteps):
        hist, _ = np.histogram(dT_cond[i, :], bins=bins_cond)
        histogram_cond_array[i, :] = hist / hist.max()
        hist, _ = np.histogram(dT_hull[i, :], bins=bins_hull)
        histogram_hull_array[i, :] = hist / hist.max()
        hist, _ = np.histogram(dT_fluid[i, :], bins=bins_fluid)
        histogram_fluid_array[i, :] = hist / hist.max()

    # Indicate bins with zero counts.
    masked_histogram_cond_array = np.ma.masked_where(histogram_cond_array == 0, histogram_cond_array)
    masked_histogram_hull_array = np.ma.masked_where(histogram_hull_array == 0, histogram_hull_array)
    masked_histogram_fluid_array = np.ma.masked_where(histogram_fluid_array == 0, histogram_fluid_array)

    # Prepare the plot.
    (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(13, 4.5))[1]

    # Create plot for conductors.
    img0 = ax0.imshow(
        masked_histogram_cond_array.T,
        aspect='auto',
        origin='lower',
        extent=[0, t_end, error_cond_min, error_cond_max],
        cmap="Blues",
        interpolation='nearest',
    )
    ax0.set_xlabel("$t$ [h]")
    ax0.set_ylabel(r"$T_\mathrm{pred} - T_\mathrm{true}$ [K]")
    ax0.set_title(f"Conductor error histogram over time")

    # Create plot for hull.
    img1 = ax1.imshow(
        masked_histogram_hull_array.T,
        aspect='auto',
        origin='lower',
        extent=[0, t_end, error_hull_min, error_hull_max],
        cmap="Blues",
        interpolation='nearest',
    )
    ax1.set_xlabel("$t$ [h]")
    ax1.set_ylabel(r"$T_\mathrm{pred} - T_\mathrm{true}$ [K]")
    ax1.set_title(f"Hull error histogram over time")

    # Create plot for hull.
    img2 = ax2.imshow(
        masked_histogram_fluid_array.T,
        aspect='auto',
        origin='lower',
        extent=[0, t_end, error_fluid_min, error_fluid_max],
        cmap="Blues",
        interpolation='nearest',
    )
    ax2.set_xlabel("$t$ [h]")
    ax2.set_ylabel(r"$T_\mathrm{pred} - T_\mathrm{true}$ [K]")
    ax2.set_title(f"Fluid error histogram over time")

    # This makes it easier to see bins with very low counts.
    if mask_out_zero_bins:
        img0.cmap.set_bad("black")
        img1.cmap.set_bad("black")
        img2.cmap.set_bad("black")

    # Finalize and show/write plot.
    plt.tight_layout()
    print(f"Writing error-time-histogram-plot to: '{error_hist_t_png}'")
    plt.savefig(error_hist_t_png, bbox_inches="tight", dpi=300)
    plt.show()