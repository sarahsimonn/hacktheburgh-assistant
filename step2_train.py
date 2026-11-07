"""
STEP 2 - Label the data, train the model, test it honestly.

What it does (in plain English):
  1. LABEL: for every drive on every day -> "did this drive fail within the next 30 days?" (yes/no)
  2. FEATURES: the SMART health readings + how much each error counter grew in the last 7 days
     + drive age + model.
  3. SPLIT BY TIME: train on the earlier dates, test on the later dates, with a 30-day gap
     so the model can never peek at the future.
  4. TRAIN a weighted gradient-boosted tree model, using each row's sampling weight.
  5. TEST it against a simple rule ("flag any drive with any error count above zero").
  6. SAVE a "fleet snapshot" - every drive on the first test day with its risk score -
     which the dashboard (step 3) uses.

Run:
    python step2_train.py --data data/drives_small.parquet

Options:
    --horizon 30      predict failure within this many days
    --split time      (default, most honest) or 'drive' (use if you only have ONE quarter of data)
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

ERROR_COUNTERS = [5, 187, 188, 197, 198, 199]  # SMART stats that count problems


def build_features(df):
    df = df.sort_values(["serial_number", "date"]).reset_index(drop=True)
    feats = []
    for i in ERROR_COUNTERS:
        col = f"smart_{i}_raw"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # How much did this error counter grow over the last 7 readings?
        df[f"d7_smart_{i}"] = df.groupby("serial_number")[col].diff(7).fillna(0)
        feats += [col, f"d7_smart_{i}"]
    for i in [12, 194]:
        col = f"smart_{i}_raw"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        feats.append(col)
    df["age_days"] = pd.to_numeric(df["smart_9_raw"], errors="coerce") / 24
    df["capacity_tb"] = pd.to_numeric(df["capacity_bytes"], errors="coerce") / 1e12
    df["model"] = df["model"].astype("category")
    feats += ["age_days", "capacity_tb", "model"]
    return df, feats


def add_labels(df, horizon):
    fail_date = df.loc[df["failure"] == 1].groupby("serial_number")["date"].min()
    df["fail_date"] = df["serial_number"].map(fail_date)
    days_to_fail = (df["fail_date"] - df["date"]).dt.days
    df["label"] = ((days_to_fail >= 0) & (days_to_fail <= horizon)).astype(int)
    # Rows in the last `horizon` days can't be labelled yet (we can't see far enough ahead).
    last_day = df["date"].max()
    df = df[df["date"] <= last_day - pd.Timedelta(days=horizon)]
    df = df[~(days_to_fail.reindex(df.index) < 0)]  # safety: drop any rows after a failure
    return df


def split(df, how, horizon):
    if how == "time":
        dates = np.sort(df["date"].unique())
        cut = pd.Timestamp(dates[int(len(dates) * 0.6)])
        train = df[df["date"] <= cut - pd.Timedelta(days=horizon)]  # gap: no peeking
        test = df[df["date"] > cut]
    else:  # split by drive: 70% of drives for training, 30% for testing
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
    df["date"] = pd.to_datetime(df["date"])
    df, feats = build_features(df)
    df = add_labels(df, args.horizon)
    train, test = split(df, args.split, args.horizon)

    print(f"Train rows: {len(train):,} (positives {train['label'].sum():,}) | "
          f"Test rows: {len(test):,} (positives {test['label'].sum():,})")
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

    # Train on the sampled data, weighting each row back to its representation
    # in the real fleet.
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
    test["risk"] = model.predict_proba(test[feats])[:, 1]
    # Simple baseline rule an engineer might use: any error counter above zero.
    err_cols = [f"smart_{i}_raw" for i in ERROR_COUNTERS]
    test["rule"] = (test[err_cols].fillna(0) > 0).any(axis=1).astype(float)
    w = test["weight"]

    metrics = {
        "horizon_days": args.horizon,
        "split": args.split,
        "test_failure_rate": float(np.average(test["label"], weights=w)),
        "model_avg_precision": float(average_precision_score(test["label"], test["risk"], sample_weight=w)),
        "rule_avg_precision": float(average_precision_score(test["label"], test["rule"], sample_weight=w)),
        "model_roc_auc": float(roc_auc_score(test["label"], test["risk"], sample_weight=w)),
        "rule_roc_auc": float(roc_auc_score(test["label"], test["rule"], sample_weight=w)),
    }

    # Fleet snapshot: every drive on the first test day -> what the dashboard shows.
    snap_day = test["date"].min()
    snap = test[test["date"] == snap_day].copy()
    keep = ["serial_number", "model", "age_days", "capacity_tb", "risk", "rule", "label", "weight"] + err_cols + ["smart_194_raw"]
    snap = snap[keep].rename(columns={"label": "failed_within_horizon"})
    snap["model"] = snap["model"].astype(str)
    metrics["snapshot_date"] = str(snap_day.date())
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
