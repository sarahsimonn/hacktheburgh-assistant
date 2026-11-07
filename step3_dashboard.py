"""
STEP 3 - The dashboard: "Which drives should we replace this week?"

Run:
    streamlit run step3_dashboard.py

It opens in your browser. Use the sliders on the left to set:
  * how much it costs to replace a drive early
  * how much a surprise failure costs (lost data, rebuild time, engineer time)
  * your maintenance budget (how many drives you can swap)
...and it shows which drives to replace and how much money/failures that saves,
compared with simple strategies an engineer might use instead.
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ERR = {5: "Reallocated sectors", 187: "Uncorrectable errors", 188: "Command timeouts",
       197: "Pending sectors", 198: "Offline uncorrectable", 199: "Cable/CRC errors"}

st.set_page_config(page_title="Drive failure predictor", layout="wide")


@st.cache_data
def load():
    snap = pd.read_parquet("data/snapshot.parquet")
    with open("data/metrics.json") as f:
        metrics = json.load(f)
    return snap, metrics


snap, metrics = load()
horizon = metrics["horizon_days"]

st.title("Which hard drives should we replace this week?")
st.caption(f"Fleet snapshot on {metrics['snapshot_date']} · predicting failure within {horizon} days · "
           f"~{metrics['snapshot_drives_estimated']:,.0f} drives in the fleet "
           f"({metrics['snapshot_drives_sampled']:,} shown) · Data: Backblaze Drive Stats")

# ---------------- Controls ----------------
with st.sidebar:
    st.header("Your costs")
    c_replace = st.slider("Cost to replace a drive early (£)", 50, 1000, 200, 25)
    c_fail = st.slider("Cost of a surprise failure (£)", 200, 20000, 3000, 100)
    budget = st.slider("Max drives you can replace this week", 0, 500, 100, 10)
    st.markdown(f"**Break-even risk:** replace a drive if its failure chance is above "
                f"**{c_replace / c_fail:.1%}** (cost to replace ÷ cost of failure).")


# ---------------- Strategies ----------------
def pick(order_score, eligible=None):
    """Pick drives in order of score until the budget (in real-fleet drives) is used."""
    df = snap.assign(score=order_score)
    if eligible is not None:
        df = df[eligible]
    df = df.sort_values("score", ascending=False)
    chosen = df[df["weight"].cumsum() <= budget]
    return set(chosen["serial_number"])


def evaluate(chosen):
    s = snap
    is_chosen = s["serial_number"].isin(chosen)
    fails = s["failed_within_horizon"] == 1
    replaced = s.loc[is_chosen, "weight"].sum()
    caught = s.loc[is_chosen & fails, "weight"].sum()
    missed = s.loc[~is_chosen & fails, "weight"].sum()
    false_alarm = s.loc[is_chosen & ~fails, "weight"].sum()
    cost = replaced * c_replace + missed * c_fail
    return {"Drives replaced": replaced, "Failures caught": caught, "Failures missed": missed,
            "Healthy drives replaced": false_alarm, "Total cost (£)": cost}


total_fail = snap.loc[snap["failed_within_horizon"] == 1, "weight"].sum()
err_cols = [f"smart_{i}_raw" for i in ERR]
err_total = snap[err_cols].fillna(0).sum(axis=1)

strategies = {
    "Our model": pick(snap["risk"], eligible=snap["risk"] * c_fail > c_replace),
    "Rule: any errors > 0": pick(err_total, eligible=err_total > 0),
    "Replace oldest first": pick(snap["age_days"].fillna(0)),
    "Do nothing": set(),
}
results = pd.DataFrame({name: evaluate(ch) for name, ch in strategies.items()}).T
baseline_cost = results.loc["Do nothing", "Total cost (£)"]
results["Saving vs doing nothing (£)"] = baseline_cost - results["Total cost (£)"]

# ---------------- Headline numbers ----------------
ours = results.loc["Our model"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Drives to replace", f"{ours['Drives replaced']:,.0f}")
c2.metric("Failures prevented", f"{ours['Failures caught']:,.0f} of {total_fail:,.0f}")
c3.metric("Healthy drives replaced", f"{ours['Healthy drives replaced']:,.0f}")
c4.metric("Money saved vs doing nothing", f"£{ours['Saving vs doing nothing (£)']:,.0f}")

# ---------------- Fleet grid ----------------
st.subheader("The fleet: every square is a drive, coloured by risk")
grid = snap.sort_values("risk", ascending=False).reset_index(drop=True)
cols = 60
grid["x"] = grid.index % cols
grid["y"] = -(grid.index // cols)
chosen = grid["serial_number"].isin(strategies["Our model"])
fig = go.Figure(go.Scatter(
    x=grid["x"], y=grid["y"], mode="markers",
    marker=dict(symbol="square", size=11, color=np.log10(grid["risk"].clip(1e-6, 1)),
                colorscale="RdYlGn_r", cmin=-5, cmax=0,
                line=dict(width=np.where(chosen, 2, 0), color="black"),
                colorbar=dict(title="Risk", tickvals=[-5, -4, -3, -2, -1, 0],
                              ticktext=["0.001%", "0.01%", "0.1%", "1%", "10%", "100%"])),
    text=[f"{r.serial_number}<br>{r.model}<br>Risk: {r.risk:.2%}<br>Age: {r.age_days:.0f} days"
          + ("<br><b>Actually failed</b>" if r.failed_within_horizon else "")
          for r in grid.itertuples()],
    hoverinfo="text",
))
fig.update_layout(height=420, xaxis_visible=False, yaxis_visible=False,
                  margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, width="stretch")
st.caption("Black outline = our model says replace it. Sorted from riskiest (top left) to safest.")

# ---------------- Strategy comparison ----------------
st.subheader("How does that compare with simpler strategies?")
st.dataframe(results.style.format("{:,.0f}"), width="stretch")

# ---------------- Replacement list with reasons ----------------
st.subheader("Replacement list: and why")
todo = snap[snap["serial_number"].isin(strategies["Our model"])].sort_values("risk", ascending=False)


def reasons(row):
    out = [f"{name}: {int(row[f'smart_{i}_raw'])}" for i, name in ERR.items()
           if pd.notna(row[f"smart_{i}_raw"]) and row[f"smart_{i}_raw"] > 0]
    return ", ".join(out) if out else "No error counts yet: flagged by age/model pattern"


if len(todo):
    show = todo.head(50).assign(
        **{"Failure chance": todo["risk"].map("{:.1%}".format),
           "Warning signs": todo.head(50).apply(reasons, axis=1),
           "Actually failed?": todo["failed_within_horizon"].map({1: "Yes", 0: "No"})})
    st.dataframe(show[["serial_number", "model", "Failure chance", "Warning signs", "Actually failed?"]],
                 width="stretch", hide_index=True)
else:
    st.info("With these costs, no drive is risky enough to be worth replacing early.")

with st.expander("Model test results (honest numbers)"):
    st.json(metrics)
