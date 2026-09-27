#!/usr/bin/env python
"""Check that the released tables and figure data reproduce from the raw records."""
import argparse
import hashlib
import json
import logging
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import _utils

# Define the directory holding this script and the other package scripts
CODE_DIRECTORY = pathlib.Path(__file__).resolve().parent

# Define the R scripts that draw the figures
PLOT_SCRIPTS = ["setup.R", "utils.R", "plots.R"]

# Define the absolute tolerance for comparing floating-point cells
ABSOLUTE_TOLERANCE = 1e-9

# Define how many characters of a failed script's output are shown
TAIL_LENGTH = 2000


################################################################################


def hash_file(path: pathlib.Path) -> str:
    """Compute the SHA-256 digest of a file."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def is_nested(value: object) -> bool:
    """Check whether a cell holds a list, dictionary, or array."""
    return isinstance(value, (list, dict, np.ndarray))


def to_json(value: object) -> str:
    """Serialize a nested cell to a canonical JSON string."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return json.dumps(value, sort_keys=True, default=str)


def normalize_table(table: pd.DataFrame) -> pd.DataFrame:
    """Sort a table's columns and rows, encoding nested cells as JSON text."""
    # Order the columns by name
    columns = sorted(table.columns)
    table = table[columns].copy()

    # Nested cells cannot be sorted, so compare them by their JSON text
    for column in columns:
        if table[column].dtype == object and table[column].map(is_nested).any():
            table[column] = table[column].map(to_json)

    # Order the rows
    return table.sort_values(columns, kind="stable").reset_index(drop=True)


def compare_tables(released: pd.DataFrame, rebuilt: pd.DataFrame) -> str | None:
    """Describe the first difference between two tables, or None if they match."""
    # The tables must have the same columns and the same number of rows
    if set(released.columns) != set(rebuilt.columns):
        columns = sorted(set(released.columns) ^ set(rebuilt.columns))
        return f"columns differ: {columns}"
    if len(released) != len(rebuilt):
        return f"row counts differ: {len(released)} vs {len(rebuilt)}"

    # Compare the tables column by column, row order aside
    released = normalize_table(released)
    rebuilt = normalize_table(rebuilt)
    for column in released.columns:
        released_column = released[column]
        rebuilt_column = rebuilt[column]
        if is_numeric_dtype(released_column) and is_numeric_dtype(rebuilt_column):
            same = np.isclose(
                released_column.to_numpy(dtype=float),
                rebuilt_column.to_numpy(dtype=float),
                atol=ABSOLUTE_TOLERANCE,
                rtol=0,
                equal_nan=True,
            )
        else:
            # None and NaN are the same missing value after a round trip
            same = (
                released_column.where(released_column.notna(), "<NA>").astype(str)
                == rebuilt_column.where(rebuilt_column.notna(), "<NA>").astype(str)
            ).to_numpy()
        if not same.all():
            row = int(np.flatnonzero(~same)[0])
            n_different = int((~same).sum())
            return (
                f"column {column!r} differs at sorted row {row} ({n_different} cells):"
                f" {released_column.iloc[row]!r} vs {rebuilt_column.iloc[row]!r}"
            )
    return None


def check_manifest(manifest_path: pathlib.Path) -> int:
    """Count the manifest files that are missing or whose digest changed."""
    # Read the manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
    n_problems = 0
    for entry in manifest["files"]:
        path = manifest_path.parent / entry["path"]
        if not path.exists() or hash_file(path) != entry["sha256"]:
            n_problems += 1
            logging.error("Manifest file %s missing or changed", path)
    logging.info("Manifest: %i files, %i problems", len(manifest["files"]), n_problems)
    return n_problems


def rebuild_tables(
    raw_directory: pathlib.Path,
    output_directory: pathlib.Path,
) -> None:
    """Rebuild the tables from the raw records into a directory."""
    # Run build_tables.py on the raw records
    result = subprocess.run(
        [
            sys.executable,
            str(CODE_DIRECTORY / "build_tables.py"),
            "--raw",
            str(raw_directory),
            "--out",
            str(output_directory),
            "--log-file",
            str(output_directory.parent / "build_tables.log"),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        print(result.stderr[-TAIL_LENGTH:], file=sys.stderr)
        print("build_tables.py failed", file=sys.stderr)
        sys.exit(1)


def check_tables(
    tables_directory: pathlib.Path,
    rebuilt_directory: pathlib.Path,
) -> int:
    """Count the released tables that the rebuilt tables do not reproduce."""
    # Compare every table that build_tables.py writes
    n_problems = 0
    for name in _utils.TABLES:
        released_path = tables_directory / f"{name}.parquet"
        rebuilt_path = rebuilt_directory / released_path.name
        if not released_path.exists():
            n_problems += 1
            logging.error("Table %s missing from the release", released_path.name)
            continue
        if not rebuilt_path.exists():
            n_problems += 1
            logging.error("Table %s not rebuilt", released_path.name)
            continue
        difference = compare_tables(
            pd.read_parquet(released_path), pd.read_parquet(rebuilt_path)
        )
        if difference is None:
            logging.info("Table %s ok", released_path.name)
        else:
            n_problems += 1
            logging.error("Table %s differs: %s", released_path.name, difference)
    return n_problems


def render_figures(
    tables_directory: pathlib.Path,
    render_directory: pathlib.Path,
) -> None:
    """Draw the figures from the tables into a directory, under both rewards."""
    # Lay out the working directory that plots.R expects
    render_directory.mkdir()
    (render_directory / "tables").symlink_to(tables_directory.resolve())
    (render_directory / "figures").mkdir()
    for script in PLOT_SCRIPTS:
        (render_directory / script).symlink_to(CODE_DIRECTORY / script)

    # Draw the figures under the free and the constrained reward
    for reward in _utils.REWARDS:
        result = subprocess.run(
            ["Rscript", "plots.R", "--reward", reward],
            cwd=render_directory,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            print(result.stderr[-TAIL_LENGTH:], file=sys.stderr)
            print(f"plots.R failed under the {reward} reward", file=sys.stderr)
            sys.exit(1)


def check_figure_data(
    figures_directory: pathlib.Path,
    rendered_directory: pathlib.Path,
) -> int:
    """Count the released figure-data files that a re-render does not reproduce."""
    # Every released figure-data file must have a rendered twin
    n_problems = 0
    released_paths = sorted(figures_directory.glob("*_data.csv"))
    if not released_paths:
        n_problems += 1
        logging.error("No figure data found in %s", figures_directory)
    for released_path in released_paths:
        rendered_path = rendered_directory / released_path.name
        if not rendered_path.exists():
            n_problems += 1
            logging.error("Figure data %s not rendered", released_path.name)
            continue
        difference = compare_tables(
            pd.read_csv(released_path), pd.read_csv(rendered_path)
        )
        if difference is None:
            logging.info("Figure data %s ok", released_path.name)
        else:
            n_problems += 1
            logging.error("Figure data %s differs: %s", released_path.name, difference)

    # Every rendered figure-data file must have been released
    for rendered_path in sorted(rendered_directory.glob("*_data.csv")):
        if not (figures_directory / rendered_path.name).exists():
            n_problems += 1
            logging.error("Figure data %s not released", rendered_path.name)
    return n_problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--raw", type=pathlib.Path, default=_utils.RAW_DIRECTORY)
    parser.add_argument("--tables", type=pathlib.Path, default=_utils.TABLES_DIRECTORY)
    parser.add_argument(
        "--figures", type=pathlib.Path, default=_utils.FIGURES_DIRECTORY
    )
    parser.add_argument("--manifest", type=pathlib.Path, default=None)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=args.log_level,
        filename=args.log_file,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # If the figures are checked, R must be available
    if not args.skip_figures and shutil.which("Rscript") is None:
        print("Rscript not found; pass --skip-figures", file=sys.stderr)
        sys.exit(1)

    # Check that every raw file listed in the manifest is present and unchanged
    n_problems = 0
    if args.manifest is not None:
        n_problems += check_manifest(args.manifest)

    with tempfile.TemporaryDirectory() as scratch:
        scratch_directory = pathlib.Path(scratch)

        # Rebuild the tables from the raw records and compare with the release
        rebuilt_directory = scratch_directory / "tables"
        rebuild_tables(args.raw, rebuilt_directory)
        n_problems += check_tables(args.tables, rebuilt_directory)

        # Redraw the figures from the released tables and compare their data
        # with the release
        if not args.skip_figures:
            render_directory = scratch_directory / "render"
            render_figures(args.tables, render_directory)
            n_problems += check_figure_data(args.figures, render_directory / "figures")

    # Report the verdict
    if n_problems:
        print(f"RESULT: FAIL ({n_problems} problems)", file=sys.stderr)
        sys.exit(1)
    print("RESULT: PACKAGE REPRODUCES")


if __name__ == "__main__":
    main()
