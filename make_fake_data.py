"""
OPTIONAL - Make FAKE data in the exact Backblaze format, so you can test the whole
pipeline in 1 minute before the real (multi-GB) download finishes.

    python make_fake_data.py
    python step1_prepare.py data/fake_Q1.zip data/fake_Q2.zip --out data/drives_small.parquet --sample 0.2

Do NOT present results from fake data - it's only for checking the code runs.
"""
import io
import zipfile

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
N_DRIVES, DAYS = 4000, 182
models = ["ST12000NM0008", "ST16000NM001G", "WDC WUH721816ALE6L4", "TOSHIBA MG08ACA16TA"]
serials = np.array([f"FAKE{i:06d}" for i in range(N_DRIVES)])
model = rng.choice(models, N_DRIVES)
age0 = rng.integers(100, 40000, N_DRIVES).astype(float)  # power-on hours at start
frailty = rng.random(N_DRIVES) < 0.06                     # some drives start degrading
smart5 = np.zeros(N_DRIVES); smart197 = np.zeros(N_DRIVES); smart187 = np.zeros(N_DRIVES)
alive = np.ones(N_DRIVES, bool)
start = pd.Timestamp("2025-01-01")
files = {"Q1": {}, "Q2": {}}

for d in range(DAYS):
    date = start + pd.Timedelta(days=d)
    grow = frailty & alive & (rng.random(N_DRIVES) < 0.15)
    smart5 += grow * rng.poisson(3, N_DRIVES)
    smart197 += grow * rng.poisson(1, N_DRIVES)
    smart187 += (frailty & alive & (rng.random(N_DRIVES) < 0.03))
    hazard = 0.00002 + 0.0006 * np.log1p(smart5 + 3 * smart197)
    fail_today = alive & (rng.random(N_DRIVES) < hazard)
    idx = np.where(alive)[0]
    df = pd.DataFrame({
        "date": date.strftime("%Y-%m-%d"), "serial_number": serials[idx], "model": model[idx],
        "capacity_bytes": 16_000_900_661_248, "failure": fail_today[idx].astype(int),
        "datacenter": "sac0", "smart_1_normalized": 100, "smart_1_raw": 0,
        "smart_5_raw": smart5[idx], "smart_9_raw": age0[idx] + 24 * d,
        "smart_12_raw": rng.integers(1, 30, len(idx)), "smart_187_raw": smart187[idx],
        "smart_188_raw": 0, "smart_194_raw": rng.integers(22, 45, len(idx)),
        "smart_197_raw": smart197[idx], "smart_198_raw": smart197[idx] // 2, "smart_199_raw": 0,
    })
    files["Q1" if d < 91 else "Q2"][date.strftime("%Y-%m-%d")] = df.to_csv(index=False)
    alive &= ~fail_today

for q, days in files.items():
    with zipfile.ZipFile(f"data/fake_{q}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for day, csv in days.items():
            z.writestr(f"data_{q}_2025/{day}.csv", csv)
    print(f"wrote data/fake_{q}.zip ({len(days)} days)")
print(f"total fake failures: {(~alive).sum()}")
