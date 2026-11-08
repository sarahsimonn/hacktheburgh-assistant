# Hard-drive failure prediction

This repository contains a small, reproducible pipeline for ranking hard drives by their probability of failing within a configurable forecast horizon. It uses Backblaze Drive Stats data, derives features from SMART attributes and drive metadata, evaluates a weighted gradient-boosted classifier against an operational baseline, and exposes the resulting fleet snapshot through a Streamlit dashboard.

The project is intended for experimentation and decision-support demonstrations. It is not a production replacement controller: the data is observational, the healthy-drive population is sampled during preparation, and the dashboard estimates costs from a single held-out snapshot.

## Repository layout

| File | Purpose |
|---|---|
| `step1_prepare.py` | Reads Backblaze quarterly ZIP archives and writes a compact, weighted Parquet dataset. |
| `step2_train.py` | Builds labels and features, trains and evaluates the model, and writes dashboard artifacts. |
| `step3_dashboard.py` | Loads the artifacts and provides risk visualization, replacement prioritization, and strategy comparison. |
| `make_fake_data.py` | Generates a small Backblaze-shaped fixture for validating the pipeline without downloading the full dataset. |
| `requirements.txt` | Python dependencies. |

## Requirements and installation

Use Python 3.12 or 3.13. Create and activate a virtual environment, then install the dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The equivalent activation command on Windows is `.venv\Scripts\activate`. The repository does not require PyCharm; all pipeline stages are command-line scripts.

Generated datasets, model files, and downloaded archives should remain outside version control. A suitable local `.gitignore` includes:

```gitignore
data/
*.zip
*.parquet
*.joblib
```

## Data preparation

Download one or more quarterly Drive Stats ZIP archives from [Backblaze's hard-drive test data page](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data), and place them in `data/`. The preparation script reads the daily CSV files directly from the archives; they do not need to be extracted.

For a quick end-to-end smoke test:

```bash
python make_fake_data.py
python step1_prepare.py \
  data/fake_Q1.zip data/fake_Q2.zip \
  --out data/drives_small.parquet \
  --sample 0.2
python step2_train.py
streamlit run step3_dashboard.py
```

For real data, retain all failed drives and a deterministic sample of healthy drives:

```bash
python step1_prepare.py \
  data/data_Q1_2025.zip data/data_Q2_2025.zip \
  --out data/drives_small.parquet
```

Important options:

* `--sample FLOAT` controls the fraction of healthy drives retained. It defaults to `0.03`; failed drives are always retained. The value is recorded indirectly through the generated `weight` column.
* `--models MODEL [...]` restricts processing to specific drive models.
* `--out PATH` selects the output Parquet path.

The output contains the base fields (`date`, `serial_number`, `model`, `capacity_bytes`, and `failure`), selected SMART raw attributes, `was_failed_drive`, and `weight`. Healthy sampled rows receive weight `1 / sample`; failed-drive rows receive weight `1.0`. Downstream training and evaluation use these weights to approximate the full fleet despite the healthy-drive sample.

## Training and evaluation

Train with the default 30-day forecast horizon and chronological split:

```bash
python step2_train.py --data data/drives_small.parquet
```

Optional arguments:

```text
--horizon DAYS       Predict failure on the observation date or within DAYS afterward.
--split {time,drive} Use a chronological split (default) or deterministic drive-level split.
--out-dir PATH       Directory for model, metrics, and snapshot artifacts (default: data).
```

The feature pipeline:

* coerces SMART values to numeric values and preserves missing measurements;
* uses raw values and seven-observation deltas for SMART attributes 5, 187, 188, 197, 198, and 199;
* includes raw SMART attributes 12 and 194;
* converts SMART attribute 9 (power-on hours) to `age_days`;
* converts `capacity_bytes` to `capacity_tb`; and
* treats `model` as a categorical feature.

The target is a drive-day label: `1` when the drive's first recorded failure is on that date or occurs within the next `horizon` days, otherwise `0`. Rows after a drive's first failure and the final `horizon` days of the dataset are excluded because their labels or future observations are not usable for forecasting.

The default time split uses the first approximately 60% of dates for training and later dates for testing, with a horizon-sized gap between them. This gap prevents training observations near the boundary from overlapping the future window represented by test observations. `--split drive` assigns all rows for a drive to one partition using a deterministic hash and is useful when only a single quarter is available, but it does not provide the same temporal generalization test.

The model is `sklearn.ensemble.HistGradientBoostingClassifier` with fixed hyperparameters and `random_state=0`. Evaluation reports weighted average precision and weighted ROC AUC for both the model and a baseline that flags any positive value in SMART error counters 5, 187, 188, 197, 198, or 199. Average precision is the primary ranking metric because failure is a rare event; accuracy would be misleading for this class distribution.

The command writes three files to `--out-dir`:

| Artifact | Contents |
|---|---|
| `model.joblib` | Trained classifier and the ordered feature list used at inference. |
| `metrics.json` | Horizon, split mode, weighted test failure rate, model metrics, baseline metrics, and snapshot metadata. |
| `snapshot.parquet` | The first test date, including risk, baseline score, labels, weights, drive metadata, and error counters for the dashboard. |

The training script prints the same test metrics and warns when either partition contains fewer than 20 positive examples. Add more quarters or change the split only after considering the effect on the evaluation design.

## Dashboard

Start the dashboard after training:

```bash
streamlit run step3_dashboard.py
```

The dashboard loads `data/snapshot.parquet` and `data/metrics.json`. It does not retrain the model or read `model.joblib`; the snapshot already contains the risk scores used for visualization and ranking.

If training uses an `--out-dir` other than `data`, copy that run's `snapshot.parquet` and `metrics.json` into `data/` before starting Streamlit, or update `DATA_DIR` in `step3_dashboard.py`. The dashboard currently uses a fixed path relative to the repository, so it does not accept an artifact-directory command-line argument.

The sidebar exposes:

* early replacement cost;
* unexpected failure cost; and
* the maximum number of drives that can be replaced.

A drive is eligible for the model strategy when `risk * failure_cost > replacement_cost`. Eligible drives are ranked by risk and selected until the weighted fleet budget is exhausted. Results are compared with the following strategies:

* the model risk ranking;
* any positive SMART error counter;
* oldest drives first; and
* no replacements.

The displayed outcomes are weighted estimates: drives replaced, failures caught or missed, healthy drives replaced, and total cost. The fleet view uses a risk-colored Plotly grid, and the replacement table includes the SMART counters that explain each recommendation.

## Reproducibility and limitations

* Backblaze archives can change across releases; record the exact archive names and download dates used for an experiment.
* The preparation sample is deterministic because healthy-drive selection is based on the serial-number CRC32. Reusing the same archives and arguments produces the same retained population.
* The default time split is preferable when the goal is to estimate performance on future dates. A random or drive-level split can produce a different question and should be reported explicitly.
* A recorded failure is not proof that a drive was healthy before the observation period, and absence of a recorded failure is not proof that the drive is safe.
* The model estimates ranking risk, not a calibrated guarantee of failure. Thresholds and costs in the dashboard are user inputs, not learned parameters.
* SMART schemas and missingness vary by manufacturer and model. Results should be checked for model-specific coverage and drift before operational use.

## Data attribution

This project uses [Backblaze Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data). Attribute Backblaze as the source when publishing results or redistributing derived analyses.
