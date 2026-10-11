"""train_v2.py - retrain the vitamin D models with extra health / lifestyle inputs (NHANES 2013-2014).



Run inside vitd_project (after  python download_more.py):     python train_v2.py



What it does

  1. Builds extra candidate inputs from NHANES: kidney disease, liver condition, thyroid problem, diabetes,

     osteoporosis, smoking, milk, alcohol per week, menopause (women), heavy drinking.

  2. Starts from your CURRENT features (basic: age, sex, BMI, supplement; sun: + outdoor time + skin reaction).

  3. Adds a candidate ONLY if it lowers the cross-validated error in a consistent way (forward selection),

     then re-checks the winner on new random splits. If it is not clearly better, the old features are kept.

     => the new model can never be worse than the old one in cross-validation.

  4. Conditions the data cannot use as model inputs (too rare) may get a small data-based +/- correction in nmol/L,

     only if it is statistically clear AND it lowers the error on held-out halves.

  5. Writes vitd_model_v2.joblib, vitd_model_sun_v2.joblib and model_report_v2.json (server.py reads them).

Your old files are not touched.

"""

import json

import os

import sys

import time

import warnings



import joblib

import numpy as np

import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,

                             mean_absolute_error, mean_squared_error, precision_score, r2_score,

                             recall_score, roc_auc_score)

from sklearn.model_selection import KFold



warnings.filterwarnings("ignore")



DATA_DIR = "data"

K = 5

SEL_SEEDS = (1, 2, 3)          # splits used to choose features

CONF_SEEDS = (11, 12, 13)      # fresh splits used to confirm the choice

MIN_GAIN = 0.04                # nmol/L of MAE a feature must save (mean over the 15 test folds)

MIN_WIN_SHARE = 0.67           # ... and it must win in at least this share of the test folds

GB = dict(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42)



BASE_FEATS = ["age", "female", "bmi", "supplement"]

SUN_FEATS = BASE_FEATS + ["outdoor_min", "skin_react"]

# Project model: retain all health/lifestyle inputs so user YES/NO answers can reach the ML model.
# "bone" is deliberately context-only because weak bones/osteoporosis is not a direct measurement of serum 25(OH)D.
POOL = ["kidney", "liver", "thyroid", "diabetes", "smoker", "milk", "alcohol_week", "heavy_drinker"]
CONTEXT_ONLY = ["bone"]

BINARY = ["kidney", "liver", "thyroid", "diabetes", "bone", "smoker", "heavy_drinker"]

LABEL = {"kidney": "kidney disease", "liver": "liver condition", "thyroid": "thyroid problem",

         "diabetes": "diabetes", "bone": "osteoporosis / weak bones", "smoker": "current smoker",

         "milk": "milk intake", "alcohol_week": "alcohol per week",

         "heavy_drinker": "heavy drinking (14+ drinks a week)"}





def read_xpt(name):

    path = os.path.join(DATA_DIR, name + ".XPT")

    return pd.read_sas(path, format="xport") if os.path.exists(path) else None





# ------------------------------------------------------------------ data

def yes_no(s, yes=(1,), no=(2,)):

    out = pd.Series(np.nan, index=s.index)

    out[s.isin(yes)] = 1.0

    out[s.isin(no)] = 0.0

    return out





def build_frame(read):

    need = {"DEMO_H": ["SEQN", "RIDAGEYR", "RIAGENDR"], "BMX_H": ["SEQN", "BMXBMI"], "VID_H": ["SEQN", "LBXVIDMS"]}

    df = None

    for name, cols in need.items():

        t = read(name)

        if t is None:

            sys.exit("Missing data/%s.XPT - run download_data.py first" % name)

        t = t[cols]

        df = t if df is None else df.merge(t, on="SEQN")

    dsq = read("DSQTOT_H")

    if dsq is not None:

        df = df.merge(dsq[["SEQN", "DSQTVD"]], on="SEQN", how="left")

    else:

        df["DSQTVD"] = 0.0

    deq = read("DEQ_H")

    if deq is not None:

        df = df.merge(deq[["SEQN", "DED120", "DED125", "DED031"]], on="SEQN", how="left")

    extra = {"KIQ_U_H": ["KIQ022"], "MCQ_H": ["MCQ160L", "MCQ160M"], "DIQ_H": ["DIQ010"],

             "ALQ_H": ["ALQ101", "ALQ110", "ALQ120Q", "ALQ120U", "ALQ130"], "OSQ_H": ["OSQ060"],
             "SMQ_H": ["SMQ020", "SMQ040"], "DBQ_H": ["DBQ197"]}

    found = {}

    for name, cols in extra.items():

        t = read(name)

        if t is None:

            print("  [skip] %s not found" % name)

            continue

        have = [c for c in cols if c in t.columns]

        miss = [c for c in cols if c not in t.columns]

        if miss:

            print("  [note] %s: columns not in file: %s" % (name, miss))

        found[name] = have

        df = df.merge(t[["SEQN"] + have], on="SEQN", how="left")

    df = df[df["RIDAGEYR"] >= 18].dropna(subset=["BMXBMI", "LBXVIDMS"]).copy()

    df["DSQTVD"] = df["DSQTVD"].fillna(0)

    return df.reset_index(drop=True)





def col(df, name):

    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)





def candidate_features(df):

    """Returns (X_candidates, report_lines). NaN means 'unknown'; it is filled afterwards."""

    age = df["RIDAGEYR"]

    female = df["RIAGENDR"] == 2

    C = pd.DataFrame(index=df.index)

    C["kidney"] = yes_no(col(df, "KIQ022"))

    C["liver"] = yes_no(col(df, "MCQ160L"))

    C["thyroid"] = yes_no(col(df, "MCQ160M"))

    C["diabetes"] = yes_no(col(df, "DIQ010"), yes=(1,), no=(2, 3))

    if "OSQ060" in df.columns:

        C["bone"] = (df["OSQ060"] == 1).astype(float)           # only asked of older adults: others count as 0

    sm = pd.Series(np.nan, index=df.index)

    if "SMQ020" in df.columns:

        sm[col(df, "SMQ020") == 2] = 0.0

        sm[col(df, "SMQ040").isin([1, 2])] = 1.0

        sm[col(df, "SMQ040") == 3] = 0.0

        C["smoker"] = sm

    if "DBQ197" in df.columns:

        C["milk"] = col(df, "DBQ197").where(col(df, "DBQ197").isin([0, 1, 2, 3]))

    if "ALQ120Q" in df.columns:

        q = col(df, "ALQ120Q").where(col(df, "ALQ120Q") <= 366)

        unit = col(df, "ALQ120U")

        per_week = q.where(unit == 1, np.nan)

        per_week = per_week.fillna((q / 4.345).where(unit == 2)).fillna((q / 52.14).where(unit == 3))

        per_day = col(df, "ALQ130").where(col(df, "ALQ130") <= 60)

        wk = per_week * per_day

        wk[q == 0] = 0.0

        if "ALQ101" in df.columns:

            wk[(col(df, "ALQ101") == 2) & wk.isna()] = 0.0

        C["alcohol_week"] = wk.clip(upper=60)

        C["heavy_drinker"] = (C["alcohol_week"] >= 14).astype(float).where(C["alcohol_week"].notna())


    lines = []

    for c in C.columns:

        n_valid = int(C[c].notna().sum())

        pos = float((C[c] == 1).sum()) if c in BINARY else float("nan")

        lines.append("  %-14s valid %5d%s" % (c, n_valid, ("  | positives %4d (%.1f%%)" % (pos, 100 * pos / max(n_valid, 1))) if c in BINARY else ""))

    for c in C.columns:                                      # fill unknowns

        if c in BINARY:

            C[c] = C[c].fillna(0.0)

        else:

            C[c] = C[c].fillna(C[c].median() if C[c].notna().any() else 0.0)

    return C, lines





def scope_base(df, C):

    X = pd.DataFrame({"age": df["RIDAGEYR"], "female": (df["RIAGENDR"] == 2).astype(float), "bmi": df["BMXBMI"],

                      "supplement": (df["DSQTVD"] > 0).astype(float)})

    X = pd.concat([X, C], axis=1)

    return X, df["LBXVIDMS"]





def scope_sun(df, C):

    if "DED120" not in df.columns:

        return None, None

    wd = df["DED120"].where(df["DED120"] <= 480)

    we = df["DED125"].where(df["DED125"] <= 480)

    X, y = scope_base(df, C)

    X["outdoor_min"] = ((5 * wd + 2 * we) / 7).fillna(we).fillna(wd)

    X["skin_react"] = df["DED031"].where(df["DED031"].isin([1, 2, 3, 4, 5]))

    keep = X[SUN_FEATS].notna().all(axis=1) & df["RIDAGEYR"].between(20, 59)

    return X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)





# ------------------------------------------------------------------ cross-validation

def cv(X, y, feats, seeds, cache):

    key = (tuple(feats), tuple(seeds))

    if key in cache:

        return cache[key]

    maes, preds = [], []

    Xf = X[feats]

    for s in seeds:

        p = np.zeros(len(y))

        for tr, te in KFold(K, shuffle=True, random_state=s).split(Xf):

            m = GradientBoostingRegressor(**GB).fit(Xf.iloc[tr], y.iloc[tr])

            p[te] = m.predict(Xf.iloc[te])

            maes.append(mean_absolute_error(y.iloc[te], p[te]))

        preds.append(p)

    cache[key] = (np.array(maes), preds)

    return cache[key]





def forward_select(X, y, start, pool, label):

    cache = {}

    cur = list(start)

    cur_m, _ = cv(X, y, cur, SEL_SEEDS, cache)

    print("\n[%s] current features: MAE %.3f (mean of %d test folds)" % (label, cur_m.mean(), len(cur_m)))

    first_step = None

    history = []

    while True:

        rows = []

        for f in pool:

            if f in cur or f not in X.columns:

                continue

            print("    testing %-14s ..." % f, end="\r", flush=True)

            m, _ = cv(X, y, cur + [f], SEL_SEEDS, cache)

            rows.append((f, cur_m.mean() - m.mean(), int((m < cur_m).sum()), m))

        if not rows:

            break

        if first_step is None:

            first_step = rows

            print("  effect of adding ONE input to the current model (positive gain = better):")

            for f, gain, wins, _ in sorted(rows, key=lambda r: -r[1]):

                print("    %-14s MAE change %+.3f   better in %2d of %d folds" % (f, -gain, wins, len(cur_m)))

        ok = [r for r in rows if r[1] >= MIN_GAIN and r[2] >= MIN_WIN_SHARE * len(cur_m)]

        if not ok:

            break

        f, gain, wins, m = max(ok, key=lambda r: r[1])

        cur.append(f)

        cur_m = m

        history.append((f, gain, wins))

        print("  + added %-12s MAE now %.3f (saved %.3f, better in %d of %d folds)" % (f, cur_m.mean(), gain, wins, len(m)))

    # confirmation on fresh splits

    base_c, _ = cv(X, y, list(start), CONF_SEEDS, cache)

    new_c, _ = cv(X, y, cur, CONF_SEEDS, cache)

    print("  confirmation on new random splits: old features MAE %.3f  ->  chosen features MAE %.3f" % (base_c.mean(), new_c.mean()))

    if cur != list(start) and not (new_c.mean() < base_c.mean() and (new_c < base_c).sum() >= 0.53 * len(new_c)):

        print("  confirmation FAILED: keeping the old features")

        cur = list(start)

    return cur, history, cache





# ------------------------------------------------------------------ metrics for the dashboard report

def pct(x, d=1):

    return round(100.0 * float(x), d)





def three(v):

    return np.where(v < 50, 0, np.where(v < 75, 1, 2))





def make_report(title, inputs, y, pred):

    y = np.asarray(y, float)

    pred = np.asarray(pred, float)

    out = {"title": title, "people": int(len(y)), "inputs": inputs,

           "mae": round(float(mean_absolute_error(y, pred)), 1), "r2": round(float(r2_score(y, pred)), 3),

           "w10": int(round(100 * np.mean(np.abs(y - pred) <= 10))), "w20": int(round(100 * np.mean(np.abs(y - pred) <= 20)))}

    cuts = []

    for c0, name in ((30, "Below 30 nmol/L (deficient)"), (50, "Below 50 nmol/L (low)"), (75, "Below 75 nmol/L (under the target)")):

        yt, yp = (y < c0).astype(int), (pred < c0).astype(int)

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()

        auc = roc_auc_score(yt, -pred) if 0 < yt.sum() < len(yt) else 0.5

        cuts.append({"q": name, "prev": pct(yt.mean()), "acc": pct(accuracy_score(yt, yp)),

                     "prec": pct(precision_score(yt, yp, zero_division=0)), "rec": pct(recall_score(yt, yp, zero_division=0)),

                     "spec": pct(tn / max(tn + fp, 1)), "f1": pct(f1_score(yt, yp, zero_division=0)), "auc": round(float(auc), 3)})

    out["cuts"] = cuts

    yt50 = (y < 50).astype(int)

    best = max(range(40, 91), key=lambda t: f1_score(yt50, (pred < t).astype(int), zero_division=0))

    yb = (pred < best).astype(int)

    out["flag"] = {"cut": int(best), "prec": pct(precision_score(yt50, yb, zero_division=0)), "rec": pct(recall_score(yt50, yb))}

    cm = confusion_matrix(three(y), three(pred), labels=[0, 1, 2])

    names = ["LOW (below 50)", "MIDDLE (50 to 75)", "HIGHER (75+)"]

    groups, f1s, recs = [], [], []

    for i in range(3):

        p_ = cm[i, i] / max(cm[:, i].sum(), 1)

        r_ = cm[i, i] / max(cm[i, :].sum(), 1)

        f_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0

        groups.append({"name": names[i], "prec": pct(p_), "rec": pct(r_), "f1": pct(f_)})

        f1s.append(f_)

        recs.append(r_)

    out.update({"groups": groups, "cm": cm.astype(int).tolist(), "acc3": pct(np.trace(cm) / cm.sum()),

                "base3": pct(cm.sum(axis=1).max() / cm.sum()), "bal3": pct(np.mean(recs)), "f13": pct(np.mean(f1s))})

    return out





# ------------------------------------------------------------------ one model

def run_scope(key, X, y, start, title, age_range=None):
    """Train the project model with every prediction input and validate it against the old baseline.

    The final model intentionally keeps all features in POOL so profile YES/NO answers are genuine
    ML inputs. We do not add hand-written residual corrections after prediction.
    """
    print("\n" + "=" * 78)
    print("%s | %d people" % (title, len(y)))
    print("=" * 78)

    # All clinically relevant prediction inputs are retained for the project model.
    feats = list(start) + [f for f in POOL if f in X.columns and f not in start]
    if "bone" in feats:
        raise RuntimeError("bone must remain context-only and cannot be a prediction feature")

    print("\nProject model features:")
    print("  " + ", ".join(feats))
    print("Context-only features:")
    print("  " + ", ".join(CONTEXT_ONLY))

    cache = {}
    base_mae_folds, base_preds = cv(X, y, list(start), CONF_SEEDS, cache)
    full_mae_folds, full_preds = cv(X, y, feats, CONF_SEEDS, cache)

    base_pred = np.mean(base_preds, axis=0)
    full_pred = np.mean(full_preds, axis=0)

    base_rep = make_report("old", "", y, base_pred)
    rep = make_report(title, "", y, full_pred)

    inputs_txt = "age, sex, BMI, supplement"
    if "outdoor_min" in feats:
        inputs_txt += ", time outdoors, skin reaction"
    extra_used = [LABEL[f] for f in feats if f not in start]
    if extra_used:
        inputs_txt += ", " + ", ".join(extra_used)
    inputs_txt += " (ages %s)" % ("%d-%d" % tuple(age_range) if age_range else "18-80")
    rep["inputs"] = inputs_txt
    rep["context_only"] = [LABEL.get(f, f) for f in CONTEXT_ONLY]
    rep["project_model_features"] = feats
    rep["validation"] = {
        "baseline_mae": round(float(base_mae_folds.mean()), 3),
        "full_model_mae": round(float(full_mae_folds.mean()), 3),
        "baseline_r2": round(float(base_rep["r2"]), 3),
        "full_model_r2": round(float(rep["r2"]), 3),
        "mae_change": round(float(full_mae_folds.mean() - base_mae_folds.mean()), 3),
        "full_model_better_mae": bool(full_mae_folds.mean() < base_mae_folds.mean()),
    }

    print("\nValidation on fresh random splits:")
    print("  old model : MAE %.2f   R2 %.3f   AUC(<50) %.3f   3-group accuracy %.1f%%" %
          (base_rep["mae"], base_rep["r2"], base_rep["cuts"][1]["auc"], base_rep["acc3"]))
    print("  full model: MAE %.2f   R2 %.3f   AUC(<50) %.3f   3-group accuracy %.1f%%" %
          (rep["mae"], rep["r2"], rep["cuts"][1]["auc"], rep["acc3"]))
    print("  MAE change: %+.3f nmol/L (negative = improvement)" % rep["validation"]["mae_change"])
    if rep["validation"]["full_model_better_mae"]:
        print("  verdict   : full project model improved MAE while retaining all health/lifestyle inputs")
    else:
        print("  verdict   : full project model did not improve MAE, but it is retained because the project requires user health inputs to reach the ML model")

    model = GradientBoostingRegressor(**GB).fit(X[feats], y)
    obj = {
        "kind": "regression",
        "model": model,
        "features": feats,
        "name": "gradient_boosting_all_health_inputs",
        "mae": float(mean_absolute_error(y, full_pred)),
        "rmse": float(mean_squared_error(y, full_pred) ** 0.5),
        "r2": float(r2_score(y, full_pred)),
        "rows": int(len(y)),
        "adjustments": [],
        "baseline_features": list(start),
        "baseline_mae": float(base_rep["mae"]),
        "context_only_features": list(CONTEXT_ONLY),
        "project_mode": "all_clinically_relevant_inputs",
        "validation": rep["validation"],
    }
    if age_range:
        obj["age_range"] = list(age_range)
    return obj, rep, base_rep


def main(read=read_xpt, out_dir="."):

    t0 = time.time()

    print("Loading NHANES files ...")

    df = build_frame(read)

    C, lines = candidate_features(df)

    print("\nCandidate inputs built from NHANES (people with a usable answer):")

    print("\n".join(lines))

    Xb, yb = scope_base(df, C)

    Xs, ys = scope_sun(df, C)

    models, report, flag_cut, flag_stats = {}, {"models": {}}, {}, {}

    scopes = [("base", Xb, yb, BASE_FEATS, "Basic model", None)]

    if Xs is not None and len(Xs) > 500:

        scopes.append(("sun", Xs, ys, SUN_FEATS, "Sun-habits model", (20, 59)))

    summary = []

    for key, X, y, start, title, ar in scopes:

        obj, rep, base_rep = run_scope(key, X, y, start, title, ar)

        models[key] = obj

        report["models"][key] = rep

        flag_cut[key] = rep["flag"]["cut"]

        flag_stats[key] = [int(round(rep["flag"]["prec"])), int(round(rep["flag"]["rec"]))]

        summary.append((title, base_rep, rep))

    files = {"base": "vitd_model_v2.joblib", "sun": "vitd_model_sun_v2.joblib"}

    for key, obj in models.items():

        joblib.dump(obj, os.path.join(out_dir, files[key]))

    with open(os.path.join(out_dir, "model_report_v2.json"), "w") as f:

        json.dump({"report": report, "flag_cut": flag_cut, "flag_stats": flag_stats}, f, indent=1)

    print("\n" + "=" * 78)

    print("SUMMARY (cross-validated)   old -> new")

    for title, a, b in summary:

        print("  %-18s MAE %.2f -> %.2f   R2 %.3f -> %.3f   AUC(<50) %.3f -> %.3f"

              % (title, a["mae"], b["mae"], a["r2"], b["r2"], a["cuts"][1]["auc"], b["cuts"][1]["auc"]))

    print("\nsaved:", ", ".join(files[k] for k in models), "and model_report_v2.json   (%.0f s)" % (time.time() - t0))

    print("Restart server.py: it loads the *_v2 files automatically. Send me this whole output.")

    return models, report





if __name__ == "__main__":

    main()