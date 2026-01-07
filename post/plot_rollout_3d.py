# Standard library.
import pickle
import os
import sys
import pickletools
from pathlib import Path

# Configure PyVista for headless/offscreen rendering before importing it.
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    os.environ.setdefault("PYVISTA_USE_OSMESA", "true")
    os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")
    os.environ.setdefault("PYVISTA_DISABLE_X11", "true")

# Third party.
from tqdm import tqdm
import matplotlib
import pyvista as pv
import numpy as np
import h5py

# These imports are for the number-hovering.
from vtkmodules.vtkCommonCore import vtkCommand
from vtkmodules.vtkInteractionWidgets import vtkHoverWidget
from vtkmodules.vtkRenderingCore import vtkCellPicker

# Constants.
COOLWARM = matplotlib.colormaps["coolwarm"]

# ==================================================================================== #
#                                      Parameters                                      #
# ==================================================================================== #

# Choose your geometry and parameters.
geometry, cluster_def = "assembly", "r=0.04m_seed=0"
par_str, step = "I=3150_T=28.7", "t0=0_dt=5min"

# geometry, cluster_def = "module_short", "r=0.02m_seed=0"
# par_str, step = "I=3150_T=28.7", "t0=0_dt=5min"

# This is the graph HDF5-file to read from.
suffix = f"{par_str}_{cluster_def}_{step}"
geometry, par_str = 'module_long', 'I=5000_T=50_r=0.04m_seed=0_t0=0_dt=5min'
rollout_hdf5 = rf"/home/azureuser/localfiles/graph-trainer-alex/jobs/baseline_5/inference/rollout_{geometry}_{par_str}.hdf5"

# Where to dump the interactive HTML exports.
rollout_path = Path(rollout_hdf5)
job_name = rollout_path.parent.parent.name  # e.g., baseline_5
plots_dir = Path(__file__).resolve().parents[1] / "results" / "plots" / job_name
plots_dir.mkdir(parents=True, exist_ok=True)
html_cond_hull = plots_dir / f"{rollout_path.stem}_cond_hull.html"
html_fluid = plots_dir / f"{rollout_path.stem}_fluid.html"

# The scalar field to visualize in the plot.
# Options: ["T_pred", "T_true", "error", "abs_error"]
field = "abs_error"

# Select the bounds of the color coding.
# Options: ["last", "overall"] or a specific interval like [0, 5]
if field in ["error", "abs_error"]:
    clim_both = [0, 2]
else:
    clim_both = "last"
clim_cond = clim_both
clim_hull = clim_cond

# Turn on/off the fluid plot.
show_fluid_plot = True

# ==================================================================================== #
#                                    Derived values                                    #
# ==================================================================================== #

OFFSCREEN = os.environ.get("PYVISTA_OFF_SCREEN", "").lower() in ("1", "true", "yes") or not os.environ.get("DISPLAY")

# Make sure we use a valid channel.
assert field in ["T_pred", "T_true", "error", "abs_error"]

# Check the given geometry name.
assert geometry in [
    "module_short",
    "module_long",
    "module_angle",
    "assembly",
    "long_short",
    "short_angle",
], f"Invalid geometry '{geometry}'"

# Check the given clim values.
for clim in [clim_cond, clim_hull]:
    if isinstance(clim, str):
        assert clim in ["last", "overall"]
    else:
        assert isinstance(clim, list)
        assert len(clim) == 2

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

# This is the pickle-file with the assembly, required for the 3D-plot.
# Note that the fixed parameter string (I=3150_T=28.7) is correct here!
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Ensure the repo root is on sys.path so unpickling can resolve `src.*` classes.
if main_dir not in sys.path:
    sys.path.append(main_dir)

graph_rated_pkl = os.path.join(main_dir, f"results/meshes/{geometry}/ClusteredMesh_{geometry}.pkl")
cluster_hdf5 = os.path.join(main_dir, f"results/meshes/{geometry}/clustering_{geometry}_{cluster_def}.hdf5")



# with open(graph_rated_pkl, "rb") as f:
#     pickletools.dis(f)

# Load the mesh-object and the clustering.
mesh = pickle.load(open(graph_rated_pkl, "rb"))
with h5py.File(cluster_hdf5, "r") as f:
    cell_clusters = f["cell_clusters"][:]

# ==================================================================================== #
#                              Read-in / preprocess data                               #
# ==================================================================================== #

# Select the rollout-file either from the Downloads-folder or from the code-dir.
print(f"\nReading rollout from file: '{rollout_hdf5}'")

# Read the information from the given hdf5-file.
with h5py.File(rollout_hdf5, "r") as f:
    node_pos = f["node_pos"][:]
    node_types = f["node_types"][:]
    if field == "T_pred":
        scalar_history = f["temperatures_pred"][:]
        label_cond = "Temperature (K, prediction)"
    elif field == "T_true":
        scalar_history = f["temperatures"][:]
        label_cond = "Temperature (K, ground truth)"
    elif field == "error":
        label_cond = "Error (K)"
        scalar_history = f["temperatures_pred"][:] - f["temperatures"][:]
    else:
        assert field == "abs_error"
        label_cond = "Absolute error (K)"
        scalar_history = np.abs(f["temperatures_pred"][:] - f["temperatures"][:])
    num_edges = len(f["edge_src"][:])
num_timesteps, num_nodes, _ = scalar_history.shape
print(f"Found {num_timesteps} timesteps for {num_nodes} nodes ({num_edges} edges)")

# Get the conductor mask.
faces_cond = mesh.get_cell_faces(np.flatnonzero(np.isin(mesh.cell_groups, [0, 2])))
pv_mesh_cond = mesh.create_face_array(faces_cond)
mesh_cond = pv.PolyData(mesh.grid_vertices, pv_mesh_cond)
mask_1 = np.isin(mesh.cell_groups[mesh.face_cells[faces_cond]], [0, 2])
mask_2 = mesh.face_cells[faces_cond] != -1
mask_cond = np.logical_and(mask_1, mask_2)
num_faces_cond = mask_cond.sum()

# Get the hull mask.
faces_hull = mesh.get_cell_faces(np.flatnonzero(mesh.cell_groups == 1))
pv_mesh_hull = mesh.create_face_array(faces_hull)
mesh_hull = pv.PolyData(mesh.grid_vertices, pv_mesh_hull)
mask_1 = mesh.cell_groups[mesh.face_cells[faces_hull]] == 1
mask_2 = mesh.face_cells[faces_hull] != -1
mask_hull = np.logical_and(mask_1, mask_2)
num_faces_hull = mask_hull.sum()

# Collect the temperature fields for each time steps.
scalar_history_cond = np.zeros((num_timesteps, num_faces_cond, 1), dtype=np.float32)
scalar_history_hull = np.zeros((num_timesteps, num_faces_hull, 1), dtype=np.float32)
for i in tqdm(range(num_timesteps), delay=0.1, desc="Collecting scalar history"):
    cell_scalars = scalar_history[i, cell_clusters, 0]
    indices_cond = mesh.face_cells[faces_cond, np.argmax(mask_cond, axis=1)]
    indices_hull = mesh.face_cells[faces_hull, np.argmax(mask_hull, axis=1)]
    scalar_history_cond[i, :, 0] = cell_scalars[indices_cond]
    scalar_history_hull[i, :, 0] = cell_scalars[indices_hull]

# Determine the bounds for the two plots.
if isinstance(clim_cond, str):
    if clim_cond == "last":
        min_val = np.min(scalar_history_cond[-1])
        max_val = np.max(scalar_history_cond[-1])
        clim_cond = [min_val, max_val]
    else:
        min_val = np.min(scalar_history_cond)
        max_val = np.max(scalar_history_cond)
        clim_cond = [min_val, max_val]
if isinstance(clim_hull, str):
    if clim_hull == "last":
        min_val = np.min(scalar_history_hull[-1])
        max_val = np.max(scalar_history_hull[-1])
        clim_hull = [min_val, max_val]
    else:
        min_val = np.min(scalar_history_hull)
        max_val = np.max(scalar_history_hull)
        clim_hull = [min_val, max_val]

# ==================================================================================== #
#                                   Create the plot                                    #
# ==================================================================================== #

# Instantiate the plotter.
pixel_width, pixel_height = 1400, 800
p = pv.Plotter(shape=(1, 2), window_size=(pixel_width, pixel_height), off_screen=OFFSCREEN)
p.add_axes(line_width=5, labels_off=False)
p.background_color = "gray"
label_hull = label_cond + " "  # For some reason these strings must be different!

# Create the initial plot for the conductors.
p.subplot(0, 0)
p.add_title("Conductors")
mesh_cond.cell_data[label_cond] = scalar_history_cond[-1]
p.add_mesh(
    mesh_cond,
    show_edges=False,
    lighting=True,
    opacity=1,
    cmap=COOLWARM,
    clim=clim_cond,
    scalar_bar_args={'title': label_cond, "position_y": 0.13},
)

# This is for the number-hovering.
picker = vtkCellPicker()
text_actor = p.add_text(
    "",
    position=(500, 500),
    color='white',
    shadow=False,
    font_size=12,
)

# Create the initial plot for the hull.
p.subplot(0, 1)
p.add_title("Hull")
mesh_hull.cell_data[label_hull] = scalar_history_hull[-1]
p.add_mesh(
    mesh_hull,
    show_edges=False,
    lighting=True,
    opacity=1,
    cmap=COOLWARM,
    clim=clim_hull,
)

def callback_hover(_widget, _):
    """Is triggered when the mouse is moved to a new location."""

    global text_actor

    # The the pixel-position from the mouse cursor.
    xx, yy = p.iren.interactor.GetEventPosition()

    pixel_width_active = p.window_size[0]

    # Get the face-index of the face the mouse is hovering above.
    renderer = p.iren.get_poked_renderer(xx, yy)
    picker.Pick(xx, yy, 0, renderer)
    face_index = picker.GetCellId()

    # Get the face's value and derive the position where to show it.
    if xx < pixel_width_active / 2:
        value = mesh_cond.cell_data[label_cond][face_index]
        p.subplot(0, 0)
        pos = (xx, yy)
    else:
        value = mesh_hull.cell_data[label_hull][face_index]
        p.subplot(0, 1)
        pos = (xx - pixel_width_active/2, yy)

    # Derive the text. When 'point_idx' is -1,
    # it is not hovering over the void.
    text = f"{value:.3f}"
    if face_index == -1:
        text = ""

    # Create the new text.
    p.remove_actor(text_actor)
    text_actor = p.add_text(
        text,
        position=pos,
        color='white',
        shadow=False,
        font_size=12,
    )
    p.render()

# Implemenet the number-hovering.
hw = vtkHoverWidget()
hw.SetInteractor(p.iren.interactor)
hw.SetTimerDuration(100)  # Time (ms) required to trigger a hover event.
hw.AddObserver(vtkCommand.TimerEvent, callback_hover)  # Start of hover.
hw.AddObserver(vtkCommand.EndInteractionEvent, callback_hover)  # Hover ended (mouse moved).
hw.EnabledOn()


def slider_callback(value):
    """Define what happens when the slider is moved to a new position."""

    # This is the time index.
    ii = int(value)

    # Update the meshes with the new temperature data
    mesh_cond.cell_data[label_cond] = scalar_history_cond[ii]
    mesh_hull.cell_data[label_hull] = scalar_history_hull[ii]

    # This will trigger a render update for the plot.
    p.render()


# Position and define the slider.
p.subplot(0, 0)
slider = p.add_slider_widget(
    slider_callback,
    [0, num_timesteps - 1],
    title="time step [min]",
    title_height=0.025,
    value=0,
    pointa=(0.17, 0.07),
    pointb=(0.956, 0.07),
    style="modern",
    fmt="%0.0f",
    interaction_event="always",
)

# Finalize and show the plot.
p.background_color = "gray"
p.add_axes(line_width=5, labels_off=False)
p.link_views()
p.camera.up = [1, 0, 0]
p.camera.azimuth = -180
p.camera.zoom(0.95)
# Move point cloud a bit upwards.
(x, y, z) = p.camera.focal_point
p.camera.focal_point = (x - 0.11, y, z)

# Export instead of (or in addition to) on-screen rendering to avoid GLX issues.
p.export_html(str(html_cond_hull))
if not OFFSCREEN:
    p.show()
p.close()

# ==================================================================================== #
#                                Create the fluid-plot                                 #
# ==================================================================================== #

if show_fluid_plot:

    # Instantiate the plotter.
    pixel_width, pixel_height = 700, 800
    p = pv.Plotter(shape=(1, 1), window_size=(pixel_width, pixel_height), off_screen=OFFSCREEN)
    p.add_axes(line_width=5, labels_off=False)
    p.background_color = "gray"

    # Create the initial plot for the fluid clusters.
    p.subplot(0, 0)
    p.add_title("Fluid clusters")
    mask_fluid_nodes = node_types == 1
    points = pv.PolyData(node_pos[mask_fluid_nodes])
    points[label_cond] = scalar_history[-1][mask_fluid_nodes]
    p.add_points(
        points,
        render_points_as_spheres=True,
        point_size=15,
        cmap="coolwarm",
        scalars=label_cond,
        clim=clim_cond,
        scalar_bar_args={'title': label_cond, "position_y": 0.13},
    )

    def slider_callback(value):
        """Define what happens when the slider is moved to a new position."""

        # This is the time index.
        ii = int(value)

        # Update the meshes with the new temperature data
        points[label_cond] = scalar_history[ii][mask_fluid_nodes]

        # This will trigger a render update for the plot.
        p.render()

    # Position and define the slider.
    p.subplot(0, 0)
    slider = p.add_slider_widget(
        slider_callback,
        [0, num_timesteps - 1],
        title="time step [min]",
        title_height=0.025,
        value=0,
        pointa=(0.17, 0.07),
        pointb=(0.956, 0.07),
        style="modern",
        fmt="%0.0f",
        interaction_event="always",
    )

    # Finalize and show the plot.
    p.add_axes(line_width=5, labels_off=False)
    p.camera.up = [1, 0, 0]
    p.camera.azimuth = -180
    p.camera.zoom(0.95)
    # Move point cloud a bit upwards.
    (x, y, z) = p.camera.focal_point
    p.camera.focal_point = (x - 0.11, y, z)
    p.export_html(str(html_fluid))
    if not OFFSCREEN:
        p.show()
    p.close()
