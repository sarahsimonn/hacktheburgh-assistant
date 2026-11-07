"""Label the prepared data, train a classifier, and evaluate it on held-out data.

The pipeline labels each drive-day according to whether the drive fails within
the configured forecast horizon, derives SMART-based features, and evaluates a
weighted gradient-boosted tree model. With the default time split, earlier
dates are used for training and later dates for testing, separated by a gap
equal to the forecast horizon. A simple error-counter rule is evaluated as a
baseline. The script also saves a fleet snapshot for the dashboard in step 3.

Run:
    python step2_train.py --data data/drives_small.parquet

Options:
    --horizon 30  Predict failure within this many days.
    --split time  Use a chronological split (the default); ``drive`` assigns
                  drives to train or test and can be useful with one quarter
                  of data.
"""
import argparse
import json
from pathlib import Path
import zlib

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ERROR_COUNTERS = [5, 187, 188, 197, 198, 199]  # SMART attributes used as error indicators.


def build_features(df):
    """Convert raw measurements into model features and return their names."""
    # Sorting is required before calculating lagged values. Without it, the
    # seven-observation difference could compare unrelated dates for the same drive.
    df = df.sort_values(["serial_number", "date"]).reset_index(drop=True)
    feats = []
    for i in ERROR_COUNTERS:
        col = f"smart_{i}_raw"
        # Backblaze SMART fields may contain blanks or non-numeric values.
        # Coercing them to NaN lets the estimator handle missing measurements.
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Difference from the value seven observations earlier for this drive.
        # A positive value indicates that the counter increased; filling the
        # initial missing lag with zero means no change is assumed until a
        # seven-observation history is available.
        df[f"d7_smart_{i}"] = df.groupby("serial_number")[col].diff(7).fillna(0)
        feats += [col, f"d7_smart_{i}"]
    for i in [12, 194]:
        col = f"smart_{i}_raw"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        feats.append(col)
    # SMART attribute 9 reports hours in service, so convert it to days.
    df["age_days"] = pd.to_numeric(df["smart_9_raw"], errors="coerce") / 24
    # Capacity is stored in bytes; use terabytes so the feature is easier to
    # interpret and remains on a manageable numeric scale.
    df["capacity_tb"] = pd.to_numeric(df["capacity_bytes"], errors="coerce") / 1e12
    # HistGradientBoostingClassifier can consume pandas categorical columns
    # when categorical_features="from_dtype" is enabled below.
    df["model"] = df["model"].astype("category")
    feats += ["age_days", "capacity_tb", "model"]
    return df, feats


def add_labels(df, horizon):
    """Add forecast labels and remove rows outside the usable observation window.

    Rows in the final forecast horizon are excluded because their future is not
    fully observed. Rows recorded after a drive's first failure are also
    excluded. Other rows remain in the data, including rows for drives with no
    recorded failure.
    label = 1: the drive fails on that date or within the next `horizon` days.
    label = 0: the drive does not fail within the next `horizon` days.
    """
    # Use the first recorded failure for each drive as the prediction target.
    # A drive can have many daily rows, but it has only one first-failure date.
    fail_date = df.loc[df["failure"] == 1].groupby("serial_number")["date"].min()
    df["fail_date"] = df["serial_number"].map(fail_date)
    days_to_fail = (df["fail_date"] - df["date"]).dt.days
    # Positive labels are drive-day rows whose failure occurs between the
    # observation date and `horizon` days afterward, inclusive. Rows for drives
    # with no known failure have NaN days_to_fail and therefore receive label 0.
    df["label"] = ((days_to_fail >= 0) & (days_to_fail <= horizon)).astype(int)
    # The final horizon days cannot be labeled because their future is unobserved.
    last_day = df["date"].max()
    df = df[df["date"] <= last_day - pd.Timedelta(days=horizon)]
    # Exclude observations recorded after a drive's first failure.
    df = df[~(days_to_fail.reindex(df.index) < 0)]
    return df


def split(df, how, horizon):
    """Split labeled rows chronologically or by a deterministic drive hash."""
    if how == "time":
        # Select a date boundary at approximately 60% of the observed period.
        # The test set contains only dates after this boundary.
        dates = np.sort(df["date"].unique())
        cut = pd.Timestamp(dates[int(len(dates) * 0.6)])
        # Keep the forecast horizon out of the training period. This prevents
        # training rows immediately before the boundary from overlapping the
        # future window represented by the first test rows.
        train = df[df["date"] <= cut - pd.Timedelta(days=horizon)]  # Leave a horizon-sized gap.
        test = df[df["date"] > cut]
    else:  # Assign approximately 70% of drives to training and 30% to testing.
        # Hashing makes the assignment deterministic while keeping all rows
        # from a drive in the same partition, preventing drive-level leakage.
        in_test = df["serial_number"].map(lambda s: zlib.crc32(s.encode()) % 10 >= 7)
        train, test = df[~in_test], df[in_test]
    return train, test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/drives_small.parquet")
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--split", choices=["time", "drive"], default="time")
    p.add_argument("--out-dir", default="data",
                   help="directory for the model, metrics, and snapshot outputs")
    args = p.parse_args()
    if args.horizon <= 0:
        p.error("--horizon must be greater than 0")

    df = pd.read_parquet(args.data)
    # Normalize dates before calculating time differences and selecting splits.
    df["date"] = pd.to_datetime(df["date"])
    df, feats = build_features(df)
    df = add_labels(df, args.horizon)
    train, test = split(df, args.split, args.horizon)

    # These are drive-day rows, not necessarily unique drives. A positive row
    # represents a drive that fails within the configured forecast horizon.
    print(f"Train rows: {len(train):,} (positives (fails within {args.horizon} days) {train['label'].sum():,}) | "
          f"Test rows: {len(test):,} (positives (fails within {args.horizon} days) {test['label'].sum():,})")
    if train["label"].sum() < 20 or test["label"].sum() < 20:
        print("WARNING: very few failures to learn from/test on. "
              "Add another quarter of data, or re-run with --split drive.")
    if len(train) == 0 or len(test) == 0:
        raise SystemExit("Train or test set is empty. Use more data or --split drive.")
    if train["label"].nunique() < 2:
        raise SystemExit(
            "Train set contains only one label class; use more data, a different split, "
            "or a different horizon."
        )
    if test["label"].nunique() < 2:
        raise SystemExit(
            "Test set contains only one label class; use more data, a different split, "
            "or a different horizon."
        )

    # The preparation step retains every failed drive but samples healthy
    # drives. The saved row weights compensate for that sampling so the model
    # learns from a distribution closer to the full fleet.
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        categorical_features="from_dtype", random_state=0,
    )
    model.fit(
        train[feats],
        train["label"],
        sample_weight=train["weight"],
    )

    test = test.copy()
    # predict_proba(... )[:, 1] is the model's estimated probability that each
    # drive-day belongs to the positive class (failure within the horizon).
    test["risk"] = model.predict_proba(test[feats])[:, 1]
    # Baseline rule: flag a drive when any selected error counter is positive.
    err_cols = [f"smart_{i}_raw" for i in ERROR_COUNTERS]
    test["rule"] = (test[err_cols].fillna(0) > 0).any(axis=1).astype(float)
    w = test["weight"]

    # Weight evaluation by the same fleet-representation weights used during
    # training. Average precision is useful for this rare-event problem;
    # ROC AUC measures ranking quality across all classification thresholds.
    metrics = {
        "horizon_days": args.horizon,
        "split": args.split,
        "test_failure_rate": float(np.average(test["label"], weights=w)),
        "model_avg_precision": float(average_precision_score(test["label"], test["risk"], sample_weight=w)),
        "rule_avg_precision": float(average_precision_score(test["label"], test["rule"], sample_weight=w)),
        "model_roc_auc": float(roc_auc_score(test["label"], test["risk"], sample_weight=w)),
        "rule_roc_auc": float(roc_auc_score(test["label"], test["rule"], sample_weight=w)),
    }

    # Save the first test day as the fleet snapshot displayed by the dashboard.
    # This gives the dashboard a single, consistently dated fleet view rather
    # than mixing rows from different dates.
    snap_day = test["date"].min()
    snap = test[test["date"] == snap_day].copy()
    keep = ["serial_number", "model", "age_days", "capacity_tb", "risk", "rule", "label", "weight"] + err_cols + ["smart_194_raw"]
    snap = snap[keep].rename(columns={"label": "failed_within_horizon"})
    snap["model"] = snap["model"].astype(str)
    metrics["snapshot_date"] = str(snap_day.date())
    # The row count is the sampled view; the sum of weights estimates the
    # number of drives represented in the full fleet.
    metrics["snapshot_drives_sampled"] = int(len(snap))
    metrics["snapshot_drives_estimated"] = float(snap["weight"].sum())

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snap.to_parquet(output_dir / "snapshot.parquet", index=False)
    joblib.dump({"model": model, "features": feats}, output_dir / "model.joblib")
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nResults on the TEST period (higher is better):")
    print(f"  Real failure rate:            {metrics['test_failure_rate']:.4%} of drive-days")
    print(f"  Avg precision  model / rule:  {metrics['model_avg_precision']:.3f} / {metrics['rule_avg_precision']:.3f}")
    print(f"  ROC AUC        model / rule:  {metrics['model_roc_auc']:.3f} / {metrics['rule_roc_auc']:.3f}")
    print(f"\nSaved {output_dir / 'model.joblib'}, {output_dir / 'metrics.json'}, "
          f"{output_dir / 'snapshot.parquet'} "
          f"(fleet snapshot for {metrics['snapshot_date']}).")


if __name__ == "__main__":
    main()
