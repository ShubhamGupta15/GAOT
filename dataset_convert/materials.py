from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from xlsx_reader import read_xlsx_sheet


_MATERIAL_RE = re.compile(
    r"^(?P<key>\w+)\s*=\s*Material\(\"(?P<name>[^\"]+)\",\s*"
    r"rho=(?P<rho>[0-9.]+),\s*cp=(?P<cp>[0-9.]+),\s*"
    r"k=(?P<k>[0-9.]+),\s*is_solid=(?P<solid>True|False)\)"
)


@dataclass(frozen=True)
class MaterialProps:
    key: str
    rho: float
    cp: float
    k: float
    is_solid: bool


@dataclass(frozen=True)
class ComponentProps:
    name: str
    material_key: str
    emissivity: float
    heat_source_vol: float
    group: str


def load_material_properties(path: Path) -> dict[str, MaterialProps]:
    """Parse material properties from the GIS material_properties.txt file."""
    materials: dict[str, MaterialProps] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _MATERIAL_RE.match(line)
        if not match:
            continue
        key = match.group("key")
        materials[key] = MaterialProps(
            key=key,
            rho=float(match.group("rho")),
            cp=float(match.group("cp")),
            k=float(match.group("k")),
            is_solid=match.group("solid") == "True",
        )
    if not materials:
        raise ValueError(f"No materials parsed from {path}")
    return materials


def load_component_properties(path: Path, sheet_name: str = "Components") -> dict[str, ComponentProps]:
    """Parse component properties from the GIS component_properties.xlsx file."""
    rows = read_xlsx_sheet(path, sheet_name)
    if not rows:
        raise ValueError(f"No rows found in {path} (sheet '{sheet_name}')")

    header = [str(item).strip() if item is not None else "" for item in rows[0]]
    col_index = {name: idx for idx, name in enumerate(header) if name}

    required = ["Name", "Material", "Emissivity", "Heat source [W/m3]", "Group"]
    missing = [name for name in required if name not in col_index]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")

    components: dict[str, ComponentProps] = {}
    for row in rows[1:]:
        name = row[col_index["Name"]] if col_index["Name"] < len(row) else None
        if name is None or str(name).strip() == "":
            continue
        material = row[col_index["Material"]] if col_index["Material"] < len(row) else None
        emissivity = row[col_index["Emissivity"]] if col_index["Emissivity"] < len(row) else 0.0
        q_vol = row[col_index["Heat source [W/m3]"]] if col_index["Heat source [W/m3]"] < len(row) else 0.0
        group = row[col_index["Group"]] if col_index["Group"] < len(row) else ""

        components[str(name)] = ComponentProps(
            name=str(name),
            material_key=str(material) if material is not None else "",
            emissivity=float(emissivity) if emissivity is not None else 0.0,
            heat_source_vol=float(q_vol) if q_vol is not None else 0.0,
            group=str(group) if group is not None else "",
        )

    if not components:
        raise ValueError(f"No components parsed from {path}")
    return components


def _find_component_match(component_name: str, component_props: dict[str, ComponentProps]) -> str:
    match = ""
    for candidate in component_props:
        if candidate in component_name and len(candidate) > len(match):
            match = candidate
    return match


def resolve_component(
    component_name: str,
    component_props: dict[str, ComponentProps],
    material_props: dict[str, MaterialProps],
) -> tuple[ComponentProps, MaterialProps]:
    """Map a component name to component + material properties."""
    lowered = component_name.lower()
    if "fluid" in lowered:
        if "external" in lowered or "air100kpa" in lowered:
            material_key = "air100kPa"
        else:
            material_key = "air750kPa"
        component = ComponentProps(
            name=component_name,
            material_key=material_key,
            emissivity=0.0,
            heat_source_vol=0.0,
            group="fluid",
        )
        material = material_props[material_key]
        return component, material

    match = _find_component_match(component_name, component_props)
    if match:
        component = component_props[match]
        material_key = component.material_key
    else:
        material_key = component_name.split("-")[0]
        if material_key not in material_props:
            material_key = "aluminum"
        component = ComponentProps(
            name=component_name,
            material_key=material_key,
            emissivity=0.5,
            heat_source_vol=0.0,
            group="none",
        )

    material = material_props.get(material_key)
    if material is None:
        material = material_props["aluminum"]
    return component, material
