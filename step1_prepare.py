"""Create a compact Parquet dataset from Backblaze Drive Stats archives.

The script reads daily CSV files directly from one or more Backblaze ZIP
archives. It makes two passes: the first identifies drives that have failed,
and the second retains all rows for those drives, a stable sample of healthy
drives, and only the columns required by later steps.

Run from the project directory with the virtual environment activated:
    python step1_prepare.py data/data_Q1_2025.zip data/data_Q2_2025.zip --out data/drives_small.parquet

Options:
    --sample 0.03  Retain 3% of healthy drives (the default). Lower values
                   reduce processing time; higher values retain more data.
    --models ...   Retain only the specified drive models.
"""
import argparse
from pathlib import Path
import zipfile
import zlib

import pandas as pd

# SMART attributes retained for feature engineering and dashboard explanations.
SMART_IDS = [
    5,    # Reallocated sectors.
    9,    # Power-on hours.
    12,   # Power-cycle count.
    187,  # Reported uncorrectable errors.
    188,  # Command timeouts.
    194,  # Drive temperature.
    197,  # Current pending sectors.
    198,  # Offline uncorrectable sectors.
    199,  # UDMA CRC errors.
]
SMART_COLS = [f"smart_{i}_raw" for i in SMART_IDS]
BASE_COLS = ["date", "serial_number", "model", "capacity_bytes", "failure"]


def list_daily_csvs(zip_paths):
    """Yield each daily CSV as a ``(zip_path, csv_name)`` pair in filename order."""
    for zp in zip_paths:
        with zipfile.ZipFile(zp) as z:
            names = [
                n for n in z.namelist()
                if n.endswith(".csv")
                and "__MACOSX" not in n
                and not n.split("/")[-1].startswith("._")
            ]
        for n in sorted(names, key=lambda s: s.split("/")[-1]):
            yield zp, n


def read_csv_from_zip(zip_path, name, wanted, required):
    """Read selected columns from one daily CSV stored inside a ZIP archive."""
    with zipfile.ZipFile(zip_path) as z, z.open(name) as f:
        df = pd.read_csv(f, usecols=lambda c: c in wanted, low_memory=False)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{zip_path}:{name} is missing required columns: {sorted(missing)}")
    df["serial_number"] = df["serial_number"].astype("string").str.strip()
    return df.dropna(subset=["serial_number"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("zips", nargs="+", help="Backblaze quarterly ZIP file(s)")
    p.add_argument("--out", default="data/drives_small.parquet")
    p.add_argument("--sample", type=float, default=0.03,
                   help="fraction of HEALTHY drives to keep (failed drives are always kept)")
    p.add_argument("--models", nargs="*", default=None,
                   help="only keep these drive models (default: all)")
    args = p.parse_args()
    if not 0 < args.sample <= 1:
        p.error("--sample must be greater than 0 and at most 1")

    files = list(list_daily_csvs(args.zips))
    if not files:
        raise SystemExit("No CSV files found inside the ZIP(s). Check the file paths.")
    print(f"Found {len(files)} daily files.")

    # ---------- Pass 1: identify failed drives ----------
    failed = {}  # Maps each failed serial number to its recorded failure date.
    for i, (zp, name) in enumerate(files, 1):
        df = read_csv_from_zip(
            zp, name, {"date", "serial_number", "failure", "model"},
            {"date", "serial_number", "failure", "model"},
        )
        if args.models:
            df = df[df["model"].isin(args.models)]
        failures = pd.to_numeric(df["failure"], errors="coerce").eq(1)
        for s, d in df.loc[failures, ["serial_number", "date"]].itertuples(index=False):
            failed[str(s)] = d
        if i % 10 == 0 or i == len(files):
            print(f"  pass 1: {i}/{len(files)} files, failures so far: {len(failed)}")

    # ---------- Pass 2: retain failed drives and a stable healthy-drive sample ----------
    threshold = int(args.sample * 10_000)
    keep_cache = {}

    def keep(serial):
        # Use one deterministic decision per drive so it is consistent across dates.
        if serial not in keep_cache:
            serial = str(serial)
            keep_cache[serial] = serial in failed or (
                zlib.crc32(serial.encode()) % 10_000
            ) < threshold
        return keep_cache[serial]

    wanted = set(BASE_COLS + SMART_COLS)
    parts = []
    for i, (zp, name) in enumerate(files, 1):
        df = read_csv_from_zip(
            zp, name, wanted,
            {"date", "serial_number", "model", "capacity_bytes", "failure"},
        )
        if args.models:
            df = df[df["model"].isin(args.models)]
        df = df[df["serial_number"].astype(str).map(keep)]
        parts.append(df)
        if i % 10 == 0 or i == len(files):
            print(f"  pass 2: {i}/{len(files)} files")

    out = pd.concat(parts, ignore_index=True)
    if out.empty:
        raise SystemExit("No rows matched the requested models and sample.")
    for c in SMART_COLS:  # Ensure the output schema is consistent across archives.
        if c not in out.columns:
            out[c] = pd.NA
    out["date"] = pd.to_datetime(out["date"])
    out["serial_number"] = out["serial_number"].astype(str)

    # Each retained row represents one drive or, for sampled healthy drives,
    # the inverse of the sampling fraction.
    out["was_failed_drive"] = out["serial_number"].isin(failed).astype(int)
    out["weight"] = out["was_failed_drive"].map({1: 1.0, 0: 1.0 / args.sample})

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    n_drives = out["serial_number"].nunique()
    print(f"\nSaved {len(out):,} rows, {n_drives:,} drives "
          f"({len(failed):,} failed) -> {output_path}")


if __name__ == "__main__":
    main()
