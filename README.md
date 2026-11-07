# Hard drive failure predictor: HRT "Best Use of Data for Predictions & Decision Making"

**One line:** hard drives warn you before they die. We learn those warning signs from hundreds of thousands of real drives, then decide which ones to replace *this week* to save the most money.

---

## Step 0: Setup (≈15 min, everyone)

1. Pull the repo in PyCharm (**Ctrl+T**).
2. Copy these files into the repo: `step1_prepare.py`, `step2_train.py`, `step3_dashboard.py`, `make_fake_data.py`, `requirements.txt`, `README.md`.
3. Make a folder called `data` in the project.
4. Add to `.gitignore` (the data is too big for GitHub):
   ```
   data/
   *.zip
   *.parquet
   *.joblib
   ```
5. In PyCharm's terminal (the prompt should start with `(.venv)`):
   ```
   pip install -r requirements.txt
   ```
   **If this fails** (your venv is Python 3.15, and some libraries may not support it yet): make a new venv with Python **3.12 or 3.13** instead (bottom-right corner → Add New Interpreter), then rerun the command.

---

## Step 1: Get the data (≈30 min, ONE person)

**Where:** https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data. Scroll to *"Downloading the Drive Stats dataset"*.

- Download **two quarters** (e.g. the two most recent). Two quarters = enough failures to train AND test honestly. Each ZIP is large, so start now.
- Put the ZIPs in the `data` folder. **Don't unzip them**: the script reads straight from the ZIP.

**While it downloads, check that the code works with fake data (1 minute):**
```
python make_fake_data.py
python step1_prepare.py data/fake_Q1.zip data/fake_Q2.zip --out data/drives_small.parquet --sample 0.2
python step2_train.py
streamlit run step3_dashboard.py
```

**Then the real thing** (swap in your real ZIP names; takes ~10–20 min):
```
python step1_prepare.py data/data_Q1_2025.zip data/data_Q2_2025.zip --out data/drives_small.parquet
```
This keeps every failed drive + 3% of healthy drives + only the ~10 useful columns → a small file.
**Share `drives_small.parquet` with the team via Google Drive/USB** so nobody else has to download the big ZIPs.

> Only got one quarter? It still works. In step 2, add `--split drive`.

---

## Step 2: Train and test the model (≈5 min to run, then iterate)

```
python step2_train.py
```
It prints how well the model does on the **test period** vs a simple rule ("flag any drive with any error > 0"):
- **Average precision:** how good the model is at putting real failures at the top of the list (higher = better). This is the key number for a rare event.
- **ROC AUC:** 0.5 = coin flip, 1.0 = perfect.

**Ideas to improve it (Person B):** add more SMART stats, longer "growth" windows (14/30 days), try different `--horizon` values (7, 14, 30 days).

---

## Step 3: The dashboard (the demo)

```
streamlit run step3_dashboard.py
```
Opens in your browser. Three sliders: **cost to replace a drive**, **cost of a surprise failure**, **budget**. It shows:
- the fleet as a grid of squares coloured green → red by risk, with outlines on the drives to replace
- failures prevented, healthy drives wasted, and money saved
- a comparison against simpler strategies: "any errors > 0", "replace oldest first", "do nothing"
- the replacement list **with reasons** ("Pending sectors: 8, Reallocated sectors: 120")

---

## Who does what

| Person | Job |
|---|---|
| **A** | Step 1: download + shrink the data, share the file. Then: slides + demo script |
| **B** | Step 2: model + honest evaluation. Try features, check results make sense |
| **C** | Step 3: dashboard polish: wording, colours, a "pick a drive → see its history" chart |
| **Everyone** | Use the AI agents to write/extend code, but **check what they produce**. |

## Timeline (finish by the deadline with time to rehearse)
- **By 8pm:** fake-data run works on everyone's laptop; real download started
- **By 10pm:** real data shrunk + first real model results
- **By midnight:** dashboard on real data
- **Next morning:** improve the model, make slides, **rehearse the demo twice**

---

## What impresses HRT judges (say these out loud)
1. **No peeking at the future:** trained on earlier dates, tested on later ones, with a 30-day gap.
2. **Rare events handled properly:** failures are ~1 in thousands of drive-days. "Predict nothing ever fails" would be 99.9% accurate and useless, so we use average precision instead of accuracy.
3. **Sampling corrected:** we kept only 3% of healthy drives for speed, then corrected probabilities and counts back to the real fleet.
4. **A real decision, not just a prediction:** replace if *failure chance × cost of failure > cost of replacing*, within a budget.
5. **Beat simple baselines:** shown against the rules an engineer would actually use.
6. **Honest about limits:** e.g. some drives die with no warning signs at all. Say so.

## Pitch (30 seconds)
"Companies like HRT run enormous computing environments. When a drive dies you lose data and time; replace drives too early and you waste money. Hard drives actually warn you before they die. We learned those warning signs from [N] real drives in Backblaze's data centres, and built a tool that tells you exactly which drives to replace this week, and how much that saves compared with the rules engineers use today."

## Data credit
Data: **Backblaze Drive Stats** (https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data). Backblaze asks that you cite them as the source, so put this on your final slide.
