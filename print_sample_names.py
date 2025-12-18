#!/usr/bin/env python3
"""
Print sample names (and indices) from a GAOT NetCDF dataset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr


def _normalize_name(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Full path to the .nc file (overrides base/name).",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=None,
        help="Base path containing the dataset .nc file.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name without extension.",
    )
    args = parser.parse_args()

    if args.dataset_path is None:
        if args.base_path is None or args.dataset_name is None:
            parser.error("Provide --dataset-path or both --base-path and --dataset-name.")
        dataset_path = args.base_path / f"{args.dataset_name}.nc"
    else:
        dataset_path = args.dataset_path

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with xr.open_dataset(dataset_path) as ds:
        if "sample" not in ds.coords:
            raise ValueError("No 'sample' coordinate found in dataset.")
        sample_names = ds.coords["sample"].values

    print(f"Found {len(sample_names)} samples in {dataset_path}")
    for idx, name in enumerate(sample_names):
        print(f"{idx}\t{_normalize_name(name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
