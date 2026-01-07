from __future__ import annotations

from pathlib import Path
import json

from xlsx_reader import read_xlsx_sheet


def load_zone_mappings(path: Path) -> dict[str, dict[str, list[int]]]:
    """Load CFD zone mappings grouped by geometry from JSON or XLSX."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r") as handle:
            data = json.load(handle)
        return {
            str(geo): {str(name): [int(z) for z in zones] for name, zones in zones_map.items()}
            for geo, zones_map in data.items()
        }
    if suffix == ".xlsx":
        return _load_zone_mappings_from_xlsx(path)
    raise ValueError(f"Unsupported zone mapping file: {path}")


def _load_zone_mappings_from_xlsx(path: Path) -> dict[str, dict[str, list[int]]]:
    rows = read_xlsx_sheet(path, "Sheet1")
    if not rows:
        raise ValueError(f"No rows found in {path}")
    header = [str(item).strip() if item is not None else "" for item in rows[0]]
    geometry_cols = [idx for idx, name in enumerate(header) if name and name != "Name"]

    zone_maps: dict[str, dict[str, list[int]]] = {header[idx]: {} for idx in geometry_cols}
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


def build_zone_to_components(zone_map: dict[str, list[int]]) -> dict[int, list[str]]:
    """Reverse a component->zones map into zone->components."""
    zone_to_components: dict[int, list[str]] = {}
    for component, zones in zone_map.items():
        for zone in zones:
            zone_to_components.setdefault(int(zone), []).append(component)
    return zone_to_components
