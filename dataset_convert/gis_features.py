from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from materials import ComponentProps, MaterialProps, resolve_component
C_FEATURE_NAMES = ["Q", "q", "k", "Cp", "cp", "cp_vol", "V", "M", "rho"]


def infer_geometry(path: Path) -> str | None:
    """Infer geometry name from a simdata path."""
    text = str(path).lower()
    for name in (
        "module_short",
        "module_long",
        "module_angle",
        "long_short",
        "short_angle",
        "assembly",
    ):
        if name in text:
            return name
    return None


def power_loss_factor(current: float | None, rated_current: float) -> float:
    if current is None:
        return 1.0
    return (current ** 2) / (rated_current ** 2)


def build_c_features(
    zones: np.ndarray,
    volumes: np.ndarray,
    geometry: str,
    current: float | None,
    zone_maps: dict[str, dict[str, list[int]]],
    component_props: dict[str, ComponentProps],
    material_props: dict[str, MaterialProps],
    rated_current: float,
    logger,
    strict: bool = True,
) -> np.ndarray:
    """Build per-node material/source features for a sample."""
    if volumes is None:
        raise ValueError("Volumes are required to compute c features.")
    component_to_zones = zone_maps.get(geometry)
    if component_to_zones is None:
        raise ValueError(f"Missing zone mapping for geometry '{geometry}'")
    plf = power_loss_factor(current, rated_current)

    zone_ids = zones.astype(np.int64)
    features = {name: np.zeros(zone_ids.shape[0], dtype=np.float32) for name in C_FEATURE_NAMES}

    assigned_mask = np.zeros(zone_ids.shape[0], dtype=bool)

    for component_name, zone_list in component_to_zones.items():
        if not zone_list:
            continue
        comp_props, mat_props = resolve_component(component_name, component_props, material_props)

        rho = mat_props.rho
        cp = mat_props.cp
        k_val = mat_props.k
        q_base = comp_props.heat_source_vol
        q_scaled = q_base * plf

        zones_array = np.asarray(zone_list, dtype=zone_ids.dtype)
        mask = np.isin(zone_ids, zones_array)
        if not np.any(mask):
            continue

        v_vals = volumes[mask]
        m_vals = rho * v_vals

        features["rho"][mask] = rho
        features["cp"][mask] = cp
        features["cp_vol"][mask] = cp * rho
        features["k"][mask] = k_val
        features["q"][mask] = q_scaled
        features["Q"][mask] = q_scaled * v_vals
        features["V"][mask] = v_vals
        features["M"][mask] = m_vals
        features["Cp"][mask] = cp * m_vals

        assigned_mask |= mask

    unassigned_zones = np.unique(zone_ids[~assigned_mask]) if not np.all(assigned_mask) else []
    if len(unassigned_zones) > 0:
        message = f"Missing zone mappings for zones: {sorted(set(int(z) for z in unassigned_zones))}"
        if strict:
            raise ValueError(message)
        logger.warning(message)

    c_array = np.stack([features[name] for name in C_FEATURE_NAMES], axis=-1)
    return c_array.astype(np.float32)
