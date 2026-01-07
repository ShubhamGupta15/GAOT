from __future__ import annotations

from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET


_CELL_REF_RE = re.compile(r"^([A-Z]+)([0-9]+)$")


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


def read_xlsx_sheet(path: Path, sheet_name: str) -> list[list[object | None]]:
    """Read a sheet from an XLSX file without external dependencies."""
    with zipfile.ZipFile(path, "r") as archive:
        shared_strings = _load_shared_strings(archive)
        sheet_path = _read_workbook_sheet_path(archive, sheet_name)
        sheet_xml = ET.fromstring(archive.read(sheet_path))

    rows: list[list[object | None]] = []
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
