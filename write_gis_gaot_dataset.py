#!/usr/bin/env python3
"""
Convert GIS CFD HDF5 outputs into a GAOT-ready NetCDF dataset with coords normalized to [-1, 1].

The script scans `/mnt/eph0/gis_data/{modules,module_pairs}` for
`simdata*.hdf5` files, pairs them with the matching geometry file in the
same `export/` directory, optionally downsamples cells, and writes
`datasets/time_dep/gis_thermal.nc` in GAOT layout:
    - u:    [sample, time, node, channel] (temperature)
    - x:    [sample, node, coord] (cell centers)
    - zone: [sample, node] (cell zone ids, optional)
    - volume: [sample, node] (cell volumes, optional)

All samples are forced to share the same node count (after downsampling)
and time axis so GAOT loaders can stack them. Coordinates are shifted to
the sample-wise center and uniformly scaled (single factor per sample,
shared across x/y/z) so that the largest spatial extent maps to [-1, 1].
Per-sample centers and scales are stored for easy inverse scaling.
"""
from __future__ import annotations

import argparse
import logging
import re
import importlib.util
from pathlib import Path
from typing import Iterable, List, Sequence

import h5py
import numpy as np

LOGGER = logging.getLogger("write_gis_gaot_dataset")


def _find_geometry(export_dir: Path) -> Path | None:
    """Pick the geometry file next to a simdata file."""
    primary = export_dir / "geometry.hdf5"
    if primary.exists():
        return primary
    candidates = sorted(export_dir.glob("geometry-*.hdf5"))
    return candidates[0] if candidates else None


def _gather_runs(root: Path, include_modules: bool, include_pairs: bool, include_assembly: bool, verbose: bool) -> List[dict]:
    """Collect simdata/geometry pairs under the requested roots."""
    if verbose:
        LOGGER.debug(
            "[gather] scanning in %s (modules=%s, module_pairs=%s, assembly=%s)",
            root,
            include_modules,
            include_pairs,
            include_assembly
        )
    runs: List[dict] = []
    targets: List[Path] = []
    if include_modules:
        targets.append(root / "modules")
    if include_pairs:
        targets.append(root / "module_pairs")
    if include_assembly:
        targets.append(root / "GIS 3-Module-Assembly ")

    for base in targets:
        if not base.exists():
            LOGGER.warning("Skipping missing base directory: %s", base)
            continue
        for sim_path in sorted(base.rglob("simdata*.hdf5")):
            export_dir = sim_path.parent
            geom_path = _find_geometry(export_dir)
            if geom_path is None:
                if verbose:
                    LOGGER.debug("[skip] geometry missing for %s", sim_path)
                continue
            record = {
                "sim": sim_path,
                "geom": geom_path,
                "rel_name": sim_path.relative_to(root).with_suffix(""),
            }
            runs.append(
                {
                    "sim": sim_path,
                    "geom": geom_path,
                    "rel_name": sim_path.relative_to(root).with_suffix(""),
                }
            )
            if verbose:
                LOGGER.debug("[gather] paired %s with %s", record["sim"], record["geom"])
    return runs


def _parse_tags(text: str) -> dict:
    """Extract current (I) and ambient (T) tags if present."""
    tags = {}
    i_match = re.search(r"I=([0-9.]+)", text)
    t_match = re.search(r"T=([0-9.]+)", text)
    if i_match:
        try:
            tags["I"] = float(i_match.group(1))
        except ValueError:
            pass
    if t_match:
        try:
            tags["T"] = float(t_match.group(1))
        except ValueError:
            pass
    return tags


def _sample_indices(
    zones: np.ndarray | None,
    target_nodes: int,
    rng: np.random.Generator,
    stratify: bool = True,
) -> np.ndarray:
    """Pick node indices (optionally stratified by zone)."""
    total = len(zones) if zones is not None else target_nodes
    if target_nodes >= total:
        return np.arange(total)

    if stratify and zones is not None:
        indices = []
        unique, counts = np.unique(zones, return_counts=True)
        for z_val, count in zip(unique, counts):
            zone_idx = np.flatnonzero(zones == z_val)
            take = max(1, int(round(target_nodes * count / total)))
            take = min(take, len(zone_idx))
            indices.append(rng.choice(zone_idx, size=take, replace=False))
        picked = np.concatenate(indices)
        if len(picked) > target_nodes:
            picked = rng.choice(picked, size=target_nodes, replace=False)
        elif len(picked) < target_nodes:
            remaining = np.setdiff1d(np.arange(total), picked, assume_unique=False)
            extra = rng.choice(remaining, size=target_nodes - len(picked), replace=False)
            picked = np.concatenate([picked, extra])
    else:
        picked = rng.choice(total, size=target_nodes, replace=False)

    rng.shuffle(picked)
    return picked


def _read_shapes(runs: Sequence[dict], verbose: bool = False) -> tuple[int, int]:
    """Find the minimum node/time counts across runs."""
    node_counts: List[int] = []
    time_counts: List[int] = []
    for idx, run in enumerate(runs, start=1):
        if verbose:
            LOGGER.debug("[shapes] (%d/%d) reading geometry %s", idx, len(runs), run["geom"])
        with h5py.File(run["geom"], "r") as geom_f:
            node_counts.append(geom_f["cell-centers"].shape[0])
        if verbose:
            LOGGER.debug("[shapes] (%d/%d) reading sim %s", idx, len(runs), run["sim"])
        with h5py.File(run["sim"], "r") as sim_f:
            temp_ds = sim_f["temperature"]
            time_counts.append(temp_ds.shape[0])
            if temp_ds.shape[1] != node_counts[-1]:
                raise ValueError(
                    f"node mismatch between {run['sim']} and geometry "
                    f"({temp_ds.shape[1]} vs {node_counts[-1]})"
                )
    min_nodes = min(node_counts)
    min_time = min(time_counts)
    if verbose:
        LOGGER.debug("[shapes] min_nodes=%d, min_time=%d", min_nodes, min_time)
    return min_nodes, min_time


def _load_sample(
    geom_path: Path,
    sim_path: Path,
    node_count: int,
    time_idx: np.ndarray,
    rng: np.random.Generator,
    stratify: bool,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load and downsample one simulation."""
    if verbose:
        LOGGER.debug("[load] geometry %s", geom_path)
    with h5py.File(geom_path, "r") as geom_f:
        centers = geom_f["cell-centers"][:]
        zones = geom_f["cell-zones"][:] if "cell-zones" in geom_f else None
        volumes = geom_f["cell-volumes"][:] if "cell-volumes" in geom_f else None

    if verbose:
        LOGGER.debug(
            "[load] sampling nodes: target=%d, available=%d", node_count, len(centers)
        )
    node_indices = _sample_indices(zones, node_count, rng, stratify=stratify)
    centers = centers[node_indices].astype(np.float32)
    zones_sel = zones[node_indices] if zones is not None else None
    volumes_sel = volumes[node_indices] if volumes is not None else None

    if verbose:
        LOGGER.debug(
            "[load] sim %s -> time_idx len=%d, nodes=%d",
            sim_path,
            len(time_idx),
            len(node_indices),
        )
    with h5py.File(sim_path, "r") as sim_f:
        temp_ds = sim_f["temperature"]
        temps = np.asarray(temp_ds[time_idx][:, node_indices, :], dtype=np.float32)

    return temps, centers, zones_sel, volumes_sel


def _write_dataset(
    output_path: Path,
    u: np.ndarray,
    x: np.ndarray,
    time_idx: np.ndarray,
    sample_names: List[str],
    zones: np.ndarray | None,
    volumes: np.ndarray | None,
    global_min_raw: np.ndarray,
    global_max_raw: np.ndarray,
    centers_raw: np.ndarray,
    scales: np.ndarray,
    append: bool = False,
    engine: str | None = None,
):
    """Serialize arrays to NetCDF with light compression.

    If append=True and the target file exists, extend along the sample
    dimension (requires netCDF4 or h5netcdf and an unlimited sample dim)."""
    import xarray as xr  # Imported here to keep CLI help working without the dependency
    time_values = time_idx.astype(np.float32)
    ds_new = xr.Dataset()
    ds_new["u"] = (("sample", "time", "node", "channel"), u)
    # x uses a dummy time axis of length 1 to satisfy vx shape [sample, 1, node, coord]
    ds_new["x"] = (("sample", "x_time", "node", "coord"), x)
    ds_new["coord_center"] = (("sample", "coord"), centers_raw.astype(np.float32))
    ds_new["coord_scale"] = (("sample",), scales.astype(np.float32))
    ds_new = ds_new.assign_coords(
        sample=("sample", sample_names),
        time=("time", time_values),
    )
    if zones is not None:
        ds_new["zone"] = (("sample", "node"), zones)
    if volumes is not None:
        ds_new["volume"] = (("sample", "node"), volumes)

    # Raw (unscaled) coordinate bounds across all samples
    ds_new.attrs["coord_min"] = global_min_raw.tolist()
    ds_new.attrs["coord_max"] = global_max_raw.tolist()
    ds_new.attrs["coord_scaling_mode"] = "uniform_max_dim_per_sample"
    ds_new.attrs["coord_scale_formula"] = "x_norm = (x_raw - coord_center) * coord_scale ; x_raw = x_norm / coord_scale + coord_center"
    ds_new.attrs["description"] = "GIS CFD temperatures downsampled for GAOT (vx mode)"

    # Pick a NetCDF backend; fall back to scipy without compression if others unavailable.
    if engine is None:
        if importlib.util.find_spec("h5netcdf"):
            engine = "h5netcdf"
        elif importlib.util.find_spec("netCDF4"):
            engine = "netcdf4"
        else:
            engine = "scipy"
            LOGGER.warning("netCDF4/h5netcdf not found; using scipy backend without compression.")

    encoding = None
    if engine in ("netcdf4", "h5netcdf"):
        # Add conservative chunk sizes to avoid oversized HDF chunks.
        u_chunk_time = min(u.shape[1], 10)
        u_chunk_nodes = min(u.shape[2], 65536)
        x_chunk_time = min(x.shape[1], 10)
        x_chunk_nodes = min(x.shape[2], 65536)
        encoding = {
            "u": {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "dtype": "float32",
                "chunksizes": (1, u_chunk_time, u_chunk_nodes, u.shape[3]),
            },
            "x": {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "dtype": "float32",
                "chunksizes": (1, x_chunk_time, x_chunk_nodes, x.shape[3]),
            },
            "zone": {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "dtype": "int32",
                "chunksizes": (1, x_chunk_nodes),
            },
            "volume": {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "dtype": "float32",
                "chunksizes": (1, x_chunk_nodes),
            },
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Append mode keeps memory bounded by extending along sample dimension.
    if append and output_path.exists():
        if engine not in ("netcdf4", "h5netcdf"):
            raise RuntimeError("Append requires netCDF4 or h5netcdf engine.")
        import xarray as xr

        with xr.open_dataset(output_path, engine=engine) as ds_old:
            # Basic compatibility checks
            if ds_old.dims["time"] != ds_new.dims["time"]:
                raise ValueError(f"time dimension mismatch: existing {ds_old.dims['time']} vs new {ds_new.dims['time']}")
            if ds_old.dims["node"] != ds_new.dims["node"]:
                raise ValueError(f"node dimension mismatch: existing {ds_old.dims['node']} vs new {ds_new.dims['node']}")
            # Update attrs with global mins/maxes without loading data
            old_min = np.asarray(ds_old.attrs.get("coord_min", global_min_raw), dtype=np.float64)
            old_max = np.asarray(ds_old.attrs.get("coord_max", global_max_raw), dtype=np.float64)
            combined_min = np.minimum(old_min, global_min_raw)
            combined_max = np.maximum(old_max, global_max_raw)
            ds_new.attrs["coord_min"] = combined_min.tolist()
            ds_new.attrs["coord_max"] = combined_max.tolist()

        # Load existing dataset into memory then close file before rewriting, to avoid open-handle truncate issues.
        with xr.open_dataset(output_path, engine=engine) as ds_old_for_concat:
            ds_old_loaded = ds_old_for_concat.load()

        # Concatenate along sample.
        ds_concat = xr.concat([ds_old_loaded, ds_new], dim="sample")
        try:
            ds_concat.to_netcdf(output_path, mode="w", encoding=encoding, engine=engine, unlimited_dims=("sample",))
        finally:
            ds_concat.close()
    else:
        try:
            ds_new.to_netcdf(output_path, encoding=encoding, engine=engine, unlimited_dims=("sample",))
        except Exception as exc:
            if engine != "scipy":
                LOGGER.warning("to_netcdf failed with engine=%s (%s); retrying with scipy/no compression", engine, exc)
                ds_new.to_netcdf(output_path, encoding=None, engine="scipy", unlimited_dims=("sample",))
            else:
                raise


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/mnt/eph0/gis_data"),
        help="Root directory containing modules/, module_pairs/ and assembly/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/azureuser/cloudfiles/code/Users/shubham.sg.gupta.ext/datasets/time_dep/gis_thermal.nc"),
        help="Output NetCDF path (relative to GAOT root by default)",
    )
    parser.add_argument("--target-nodes", type=int, default=2048576, help="Nodes to keep per sample")
    parser.add_argument("--time-stride", type=int, default=1, help="Stride over CFD timesteps")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of simulations")
    parser.add_argument("--sample-offset", type=int, default=0, help="Skip the first N simulations (for batching)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for downsampling")
    parser.add_argument(

        "--no-stratify",
        action="store_true",
        help="Disable zone-stratified node sampling (random uniform instead)",
    )
    parser.add_argument(
        "--skip-modules", action="store_true", help="Ignore /modules runs under the source root"
    )
    parser.add_argument(
        "--skip-module-pairs",
        action="store_true",
        help="Ignore /module_pairs runs under the source root",
    )
    parser.add_argument(
        "--skip-assembly",
        action="store_true",
        help="Ignore /assembly runs under the source root",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing NetCDF (process batches separately, requires netCDF4/h5netcdf)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["netcdf4", "h5netcdf", "scipy"],
        default=None,
        help="NetCDF engine to use (default: auto, prefers h5netcdf)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress info")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    runs_all = _gather_runs(
        args.source,
        include_modules=not args.skip_modules,
        include_pairs=not args.skip_module_pairs,
        include_assembly = not args.skip_assembly,
        verbose=args.verbose,
    )
    if not runs_all:
        LOGGER.warning("No simdata files found. Check the source path and flags.")
        return 1
    if args.sample_offset:
        runs_all = runs_all[args.sample_offset :]
    if args.max_samples is not None:
        runs_all = runs_all[: args.max_samples]
    runs = runs_all

    if args.verbose:
        LOGGER.debug("Found %d simdata files after filtering", len(runs))

    # Determine node/time dims: use existing file when appending, otherwise scan the current batch.
    if args.append and Path(args.output).exists():
        import xarray as xr
        with xr.open_dataset(args.output, engine=args.engine) as ds_existing:
            # Enforce same node/time dimensions as existing file
            node_count = int(ds_existing.dims["node"])
            time_len = int(ds_existing.dims["time"])
            time_values = np.asarray(ds_existing["time"].values)
            if time_values.shape[0] != time_len:
                raise ValueError(
                    f"time coordinate length mismatch: dims={time_len}, coord={time_values.shape[0]}"
                )
            if not np.allclose(time_values, np.round(time_values), atol=1e-6):
                raise ValueError(
                    "Existing time coordinate is non-integer; cannot use as HDF5 indices when appending."
                )
            time_idx = time_values.astype(np.int64)
            min_nodes = node_count
            min_time = time_len
        if args.verbose:
            LOGGER.debug("[shapes] using existing file dims node=%d, time_len=%d for append", node_count, time_len)
    else:
        if args.verbose:
            LOGGER.debug("[shapes] start computing min node/time across batch runs")
        min_nodes, min_time = _read_shapes(runs, verbose=args.verbose)
        node_count = min(args.target_nodes, min_nodes)
        time_idx = np.arange(0, min_time, args.time_stride, dtype=np.int64)
        if len(time_idx) == 0:
            raise ValueError("time_stride is too large for available CFD timesteps.")

    if args.verbose:
        LOGGER.debug("Processing %d runs in this batch (offset=%d, max=%s)", len(runs), args.sample_offset, args.max_samples)

    rng = np.random.default_rng(args.seed)
    u_list: List[np.ndarray] = []
    x_list: List[np.ndarray] = []
    zone_list: List[np.ndarray] = []
    vol_list: List[np.ndarray] = []
    sample_names: List[str] = []

    global_min = np.full(3, np.inf, dtype=np.float64)
    global_max = np.full(3, -np.inf, dtype=np.float64)
    center_list: List[np.ndarray] = []
    scale_list: List[float] = []

    if args.verbose:
        LOGGER.debug(
            "Using node_count=%d (min available %d), time steps per sample=%d (stride %d)",
            node_count,
            min_nodes,
            len(time_idx),
            args.time_stride,
        )

    for idx, run in enumerate(runs, start=1):
        temps, centers, zones_sel, volumes_sel = _load_sample(
            run["geom"],
            run["sim"],
            node_count=node_count,
            time_idx=time_idx,
            rng=rng,
            stratify=not args.no_stratify,
            verbose=args.verbose,
        )
        if args.verbose:
            LOGGER.debug(
                "[%d/%d] loaded %s (nodes %d, times %d) before normalization",
                idx,
                len(runs),
                run["rel_name"],
                centers.shape[0],
                temps.shape[0],
            )

        raw_min = centers.min(axis=0)
        raw_max = centers.max(axis=0)
        global_min = np.minimum(global_min, raw_min)
        global_max = np.maximum(global_max, raw_max)

        # Uniform scaling per sample using the largest dimension extent
        center = 0.5 * (raw_min + raw_max)
        extent = raw_max - raw_min
        max_extent = np.max(extent)
        if max_extent <= 0:
            raise ValueError(f"Non-positive spatial extent for sample {run['rel_name']}: {extent}")
        scale = 2.0 / max_extent  # maps max dimension to [-1, 1]
        centers_norm = (centers - center) * scale

        center_list.append(center.astype(np.float32))
        scale_list.append(np.float32(scale))

        u_list.append(temps)
        x_list.append(centers_norm.astype(np.float32))
        if zones_sel is not None:
            zone_list.append(zones_sel.astype(np.int32))
        if volumes_sel is not None:
            vol_list.append(volumes_sel.astype(np.float32))
        sample_names.append(str(run["rel_name"]))

    u = np.stack(u_list, axis=0)
    x = np.stack(x_list, axis=0)
    # Expand coordinates to [sample, 1, node, coord] (geometry fixed over time)
    x = x[:, None, :, :]
    centers_raw = np.stack(center_list, axis=0)
    scales = np.stack(scale_list, axis=0)
    zones = np.stack(zone_list, axis=0) if zone_list else None
    volumes = np.stack(vol_list, axis=0) if vol_list else None

    if args.verbose:
        LOGGER.debug("[write] stacking arrays and writing NetCDF")
    _write_dataset(
        args.output,
        u=u,
        x=x,
        time_idx=time_idx,
        sample_names=sample_names,
        zones=zones,
        volumes=volumes,
        global_min_raw=global_min,
        global_max_raw=global_max,
        centers_raw=centers_raw,
        scales=scales,
        append=args.append,
        engine=args.engine,
    )

    tags = [_parse_tags(name) for name in sample_names]
    currents = [t.get("I") for t in tags if "I" in t]
    ambients = [t.get("T") for t in tags if "T" in t]

    LOGGER.info("Wrote %d samples -> %s", u.shape[0], args.output)
    LOGGER.info("  time steps: %d (stride %d)", u.shape[1], args.time_stride)
    LOGGER.info("  nodes per sample: %d", u.shape[2])
    if currents:
        LOGGER.info("  current range (I): %s .. %s", min(currents), max(currents))
    if ambients:
        LOGGER.info("  ambient range (T): %s .. %s", min(ambients), max(ambients))
    LOGGER.info("  coord bounds: min %s, max %s", global_min, global_max)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
