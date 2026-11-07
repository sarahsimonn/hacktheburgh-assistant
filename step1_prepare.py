"""
STEP 1 - Shrink the Backblaze Drive Stats data into one small file.

What it does (in plain English):
  * Reads the daily CSV files straight out of the Backblaze ZIP(s) - no unzipping needed.
  * Pass 1: finds every drive that FAILED (failure == 1) and the date it failed.
  * Pass 2: keeps ALL rows for failed drives + a small random sample of healthy drives,
    and only the ~10 columns we actually need.
  * Saves everything to ONE small Parquet file you can share with your team.

Run it like this (from the project folder, with your venv active):
    python step1_prepare.py data/data_Q1_2025.zip data/data_Q2_2025.zip --out data/drives_small.parquet

Optional:
    --sample 0.03   keep 3% of healthy drives (default). Lower = faster, higher = more data.
    --models ST12000NM0008 ST16000NM001G   only keep these drive models
"""
import argparse
import zipfile
import zlib

import pandas as pd

# SMART stats we keep. Each is a health sensor the drive reports about itself.
SMART_IDS = [
    5,    # Reallocated sectors - bad spots the drive has had to swap out
    9,    # Power-on hours - how old the drive is (in use)
    12,   # Power cycle count - how many times it has been switched on/off
    187,  # Reported uncorrectable errors
    188,  # Command timeouts
    194,  # Temperature
    197,  # Current pending sectors - spots waiting to be checked/swapped
    198,  # Offline uncorrectable sectors
    199,  # UDMA CRC errors - usually cable/connection problems
]
SMART_COLS = [f"smart_{i}_raw" for i in SMART_IDS]
BASE_COLS = ["date", "serial_number", "model", "capacity_bytes", "failure"]


def list_daily_csvs(zip_paths):
    """Yield (zip_path, csv_name) for every daily CSV inside the ZIPs, in date order."""
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


def read_csv_from_zip(zip_path, name, wanted):
    """Read one daily CSV from inside the ZIP, only loading the columns we want."""
    with zipfile.ZipFile(zip_path) as z, z.open(name) as f:
        return pd.read_csv(f, usecols=lambda c: c in wanted, low_memory=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("zips", nargs="+", help="Backblaze quarterly ZIP file(s)")
    p.add_argument("--out", default="data/drives_small.parquet")
    p.add_argument("--sample", type=float, default=0.03,
                   help="fraction of HEALTHY drives to keep (failed drives are always kept)")
    p.add_argument("--models", nargs="*", default=None,
                   help="only keep these drive models (default: all)")
    args = p.parse_args()

    files = list(list_daily_csvs(args.zips))
    if not files:
        raise SystemExit("No CSV files found inside the ZIP(s). Check the file paths.")
    print(f"Found {len(files)} daily files.")

    # ---------- Pass 1: which drives failed, and when? ----------
    failed = {}  # serial_number -> failure date
    for i, (zp, name) in enumerate(files, 1):
        df = read_csv_from_zip(zp, name, {"date", "serial_number", "failure", "model"})
        if args.models:
            df = df[df["model"].isin(args.models)]
        for s, d in df.loc[df["failure"] == 1, ["serial_number", "date"]].itertuples(index=False):
            failed[s] = d
        if i % 10 == 0 or i == len(files):
            print(f"  pass 1: {i}/{len(files)} files, failures so far: {len(failed)}")

    # ---------- Pass 2: keep failed drives + a sample of healthy ones ----------
    threshold = int(args.sample * 10_000)
    keep_cache = {}

    def keep(serial):
        # Same drive always gets the same decision on every day (stable hash, not random).
        if serial not in keep_cache:
            keep_cache[serial] = serial in failed or (zlib.crc32(serial.encode()) % 10_000) < threshold
        return keep_cache[serial]

    wanted = set(BASE_COLS + SMART_COLS)
    parts = []
    for i, (zp, name) in enumerate(files, 1):
        df = read_csv_from_zip(zp, name, wanted)
        if args.models:
            df = df[df["model"].isin(args.models)]
        df = df[df["serial_number"].astype(str).map(keep)]
        parts.append(df)
        if i % 10 == 0 or i == len(files):
            print(f"  pass 2: {i}/{len(files)} files")

    out = pd.concat(parts, ignore_index=True)
    for c in SMART_COLS:  # make sure every SMART column exists, even if a quarter lacked it
        if c not in out.columns:
            out[c] = pd.NA
    out["date"] = pd.to_datetime(out["date"])
    out["serial_number"] = out["serial_number"].astype(str)

    # Weight = how many real drives each kept row stands for.
    # Failed drives are all kept (weight 1); healthy drives were sampled (weight 1/sample).
    out["was_failed_drive"] = out["serial_number"].isin(failed).astype(int)
    out["weight"] = out["was_failed_drive"].map({1: 1.0, 0: 1.0 / args.sample})

    out.to_parquet(args.out, index=False)
    n_drives = out["serial_number"].nunique()
    print(f"\nSaved {len(out):,} rows, {n_drives:,} drives "
          f"({len(failed):,} failed) -> {args.out}")


if __name__ == "__main__":
    main()
