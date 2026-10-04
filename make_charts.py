"""make_charts.py - report charts for both models (honest, cross-validated predictions)
Needs matplotlib:  pip install matplotlib
Run inside vitd_project:  python make_charts.py      -> creates the folder  charts/
"""
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import KFold, cross_val_predict

OUT = "charts"
os.makedirs(OUT, exist_ok=True)
FLAG_CUT = {"base": 64, "sun": 62}


def load_data():
    rd = lambda f: pd.read_sas("data/%s.XPT" % f, format="xport")
    d = rd("DEMO_H")[["SEQN", "RIDAGEYR", "RIAGENDR"]].merge(rd("BMX_H")[["SEQN", "BMXBMI"]], on="SEQN") \
        .merge(rd("VID_H")[["SEQN", "LBXVIDMS"]], on="SEQN") \
        .merge(rd("DSQTOT_H")[["SEQN", "DSQTVD"]], on="SEQN", how="left")
    d = d.dropna(subset=["BMXBMI", "LBXVIDMS"])
    d["DSQTVD"] = d["DSQTVD"].fillna(0)
    d = d[d["RIDAGEYR"] >= 18]
    s = d.merge(rd("DEQ_H")[["SEQN", "DED120", "DED125", "DEQ034D", "DED031"]], on="SEQN")
    wd = s["DED120"].where(s["DED120"] <= 480)
    we = s["DED125"].where(s["DED125"] <= 480)

    def feats(t, sun):
        X = pd.DataFrame({"age": t["RIDAGEYR"], "female": (t["RIAGENDR"] == 2).astype(float),
                          "bmi": t["BMXBMI"], "supplement": (t["DSQTVD"] > 0).astype(float)})
        if sun:
            X["outdoor_min"] = ((5 * wd + 2 * we) / 7).fillna(we).fillna(wd)
            X["skin_react"] = t["DED031"].where(t["DED031"].isin([1, 2, 3, 4, 5]))
        return X

    out = {}
    for key, fn, t, sun in (("base", "vitd_model.joblib", d, False), ("sun", "vitd_model_sun.joblib", s, True)):
        m = joblib.load(fn)
        X, y = feats(t, sun), t["LBXVIDMS"]
        keep = X.notna().all(axis=1)
        if sun:
            keep &= t["RIDAGEYR"].between(20, 59)
        out[key] = (m, X[keep][m["features"]], y[keep])
    return out


def honest_pred(m, X, y):
    p = [cross_val_predict(clone(m["model"]), X, y, cv=KFold(5, shuffle=True, random_state=s)) for s in (1, 2, 3)]
    return np.mean(p, axis=0)


data = load_data()
NAMES = {"base": "Basic model (age 18-80, 4 inputs)", "sun": "Sun model (age 20-59, 6 inputs)"}
RES = {k: (np.asarray(y), honest_pred(m, X, y)) for k, (m, X, y) in data.items()}

# ---- 1. predicted vs actual
fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
for a, (k, (y, p)) in zip(ax, RES.items()):
    hb = a.hexbin(p, y, gridsize=35, cmap="Oranges", mincnt=1, extent=(25, 100, 0, 160))
    a.plot([25, 100], [25, 100], "k--", lw=1, label="perfect prediction")
    bins = pd.qcut(p, 10, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True).mean()
    a.plot(g["p"], g["y"], "o-", color="#1d4ed8", lw=2, label="average actual (10 groups)")
    a.set_title(NAMES[k])
    a.set_xlabel("Predicted 25(OH)D (nmol/L)")
    a.grid(alpha=.3)
ax[0].set_ylabel("Measured 25(OH)D (nmol/L)")
ax[0].legend(loc="upper left")
fig.suptitle("Predicted vs measured vitamin D (cross-validated, NHANES 2013-2014)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "1_predicted_vs_actual.png"), dpi=160)
plt.close(fig)

# ---- 2. ROC curves
fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
for a, cut in zip(ax, (50, 75)):
    for k, col in (("base", "#ea580c"), ("sun", "#1d4ed8")):
        y, p = RES[k]
        yt = (y < cut).astype(int)
        fpr, tpr, _ = roc_curve(yt, -p)
        a.plot(fpr, tpr, color=col, lw=2, label="%s  (AUC %.2f)" % (NAMES[k].split(" (")[0], roc_auc_score(yt, -p)))
    a.plot([0, 1], [0, 1], "k--", lw=1, label="guessing (AUC 0.50)")
    a.set_title("Real level below %d nmol/L?" % cut)
    a.set_xlabel("False positive rate")
    a.set_ylabel("True positive rate (recall)")
    a.legend(loc="lower right")
    a.grid(alpha=.3)
fig.suptitle("ROC curves (cross-validated)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "2_roc_curves.png"), dpi=160)
plt.close(fig)

# ---- 3. why the 'higher risk' flag uses 62-64 and not 50
fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for a, (k, (y, p)) in zip(ax, RES.items()):
    yt = (y < 50).astype(int)
    cuts = np.arange(40, 81)
    pr = [precision_score(yt, (p < c).astype(int), zero_division=0) for c in cuts]
    rc = [recall_score(yt, (p < c).astype(int)) for c in cuts]
    a.plot(cuts, rc, color="#16a34a", lw=2, label="recall (low people caught)")
    a.plot(cuts, pr, color="#dc2626", lw=2, label="precision (flagged who are really low)")
    a.axvline(50, color="gray", ls=":", label="cut-off 50 (old label)")
    a.axvline(FLAG_CUT[k], color="k", ls="--", label="flag at predicted < %d" % FLAG_CUT[k])
    a.set_title(NAMES[k])
    a.set_xlabel("Flag people whose predicted level is below ...")
    a.grid(alpha=.3)
ax[0].set_ylabel("Score")
ax[0].legend(loc="center right", fontsize=8)
fig.suptitle("Flagging people really below 50 nmol/L: precision and recall by flag cut-off")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "3_flag_cutoff.png"), dpi=160)
plt.close(fig)

# ---- 4. dashboard categories: confusion matrix (% of each real group)
cat3 = lambda v: np.where(v < 50, 0, np.where(v < 75, 1, 2))
labs = ["LOW\n(<50)", "MID\n(50-75)", "75+"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for a, (k, (y, p)) in zip(ax, RES.items()):
    cm = confusion_matrix(cat3(y), cat3(p))
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    a.imshow(pct, cmap="Oranges", vmin=0, vmax=100)
    for i in range(3):
        for j in range(3):
            a.text(j, i, "%.0f%%\n(%d)" % (pct[i, j], cm[i, j]), ha="center", va="center", fontsize=10)
    a.set_xticks(range(3)); a.set_xticklabels(labs)
    a.set_yticks(range(3)); a.set_yticklabels(labs)
    a.set_xlabel("Model says"); a.set_ylabel("Really is")
    a.set_title(NAMES[k])
fig.suptitle("Dashboard level groups: what the model says vs what the blood test showed")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "4_level_groups.png"), dpi=160)
plt.close(fig)

print("saved 4 charts in the folder:", os.path.abspath(OUT))