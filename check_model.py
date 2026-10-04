"""check_model.py - sanity checks for vitd_model.joblib and vitd_model_sun.joblib
Run inside vitd_project (next to the .joblib files, server.py and the data folder):
    python check_model.py
"""
import os
import joblib
import numpy as np
import pandas as pd

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print("[%s] %s %s" % (PASS if ok else FAIL, name, detail))


MODELS = {}
for key, fn in (("base", "vitd_model.joblib"), ("sun", "vitd_model_sun.joblib")):
    if os.path.exists(fn):
        MODELS[key] = joblib.load(fn)
    else:
        print("[SKIP] %s not found" % fn)

# ---------------------------------------------------------------- 1. what is inside
print("\n=== 1. Saved models ===")
for key, m in MODELS.items():
    print(key, "|", m["name"], "| features:", m["features"], "| MAE %.2f RMSE %.2f R2 %.3f rows %s"
          % (m["mae"], m["rmse"], m["r2"], m.get("rows")))
    check(key + ": kind is regression", m.get("kind") == "regression")
    check(key + ": R2 between 0 and 0.5 (NHANES with so few features)", 0 < m["r2"] < 0.5, "R2=%.3f" % m["r2"])

# ---------------------------------------------------------------- 2. test on real NHANES rows
print("\n=== 2. Predicted vs actual on NHANES rows ===")
try:
    rd = lambda f: pd.read_sas("data/%s.XPT" % f, format="xport")
    df = rd("DEMO_H")[["SEQN", "RIDAGEYR", "RIAGENDR"]].merge(rd("BMX_H")[["SEQN", "BMXBMI"]], on="SEQN") \
        .merge(rd("VID_H")[["SEQN", "LBXVIDMS"]], on="SEQN") \
        .merge(rd("DEQ_H")[["SEQN", "DED120", "DED125", "DEQ034D", "DED031"]], on="SEQN") \
        .merge(rd("DSQTOT_H")[["SEQN", "DSQTVD"]], on="SEQN", how="left")
    df = df.dropna(subset=["BMXBMI", "LBXVIDMS"])
    df["DSQTVD"] = df["DSQTVD"].fillna(0)
    wd = df["DED120"].where(df["DED120"] <= 480)
    we = df["DED125"].where(df["DED125"] <= 480)
    X = pd.DataFrame({
        "age": df["RIDAGEYR"], "female": (df["RIAGENDR"] == 2).astype(float), "bmi": df["BMXBMI"],
        "supplement": (df["DSQTVD"] > 0).astype(float),
        "outdoor_min": ((5 * wd + 2 * we) / 7).fillna(we).fillna(wd),
        "sunscreen": df["DEQ034D"].where(df["DEQ034D"].isin([1, 2, 3, 4, 5])),
        "skin_react": df["DED031"].where(df["DED031"].isin([1, 2, 3, 4, 5])),
    })
    keep = X.notna().all(axis=1)
    X, y = X[keep], df.loc[keep, "LBXVIDMS"]
    pred = pd.Series(MODELS["sun"]["model"].predict(X[MODELS["sun"]["features"]]), index=y.index)
    mae = float(np.mean(np.abs(y - pred)))
    base = float(np.mean(np.abs(y - y.mean())))
    check("sun model beats 'always predict the average'", mae < base, "MAE %.1f vs baseline %.1f" % (mae, base))
    print("rows:", len(y), "| correlation(pred, actual) = %.2f" % np.corrcoef(pred, y)[0, 1])
    dec = pd.qcut(pred, 5, duplicates="drop")
    tab = pd.DataFrame({"pred": pred, "actual": y}).groupby(dec, observed=True).mean().round(1)
    print("\nCalibration: people grouped by predicted level (5 groups). Actual should rise with predicted:")
    print(tab.to_string())
    check("calibration: actual level rises with predicted level", tab["actual"].is_monotonic_increasing)
    check("outputs are in a realistic range (10-200 nmol/L)", pred.min() > 10 and pred.max() < 200,
          "min %.0f max %.0f" % (pred.min(), pred.max()))
except Exception as e:
    print("[SKIP] NHANES test not run:", e)

# ---------------------------------------------------------------- 3. change one input at a time
print("\n=== 3. Change one input, keep the rest fixed ===")
ref = {"age": 30, "female": 1.0, "bmi": 24.0, "supplement": 0.0,
       "outdoor_min": 60.0, "sunscreen": 3.0, "skin_react": 3.0}
SWEEPS = {  # feature: (values, expected direction from NHANES / research)
    "supplement": ([0, 1], +1), "bmi": ([18, 22, 26, 30, 35, 40], -1),
    "age": ([20, 30, 40, 50, 59], +1), "female": ([0, 1], +1),
    "outdoor_min": ([0, 30, 60, 120, 240, 480], +1), "skin_react": ([1, 2, 3, 4, 5], -1),
}
for key, m in MODELS.items():
    print("\n-- model:", key)
    for f, (vals, want) in SWEEPS.items():
        if f not in m["features"]:
            print("   %-12s (not a feature of this model)" % f)
            continue
        out = [float(m["model"].predict(pd.DataFrame([dict(ref, **{f: v})])[m["features"]])[0]) for v in vals]
        net = out[-1] - out[0]
        print("   %-12s %s -> %s" % (f, vals, [round(o, 1) for o in out]))
        check("%s/%s: overall direction %s" % (key, f, "up" if want > 0 else "down"), net * want > 0,
              "(net %+.1f nmol/L)" % net)

# ---------------------------------------------------------------- 4. extreme inputs
print("\n=== 4. Extreme inputs ===")
for key, m in MODELS.items():
    lo, hi = (20, 59) if key == "sun" else (18, 80)
    rows = [dict(ref, age=a, bmi=b, outdoor_min=o, supplement=s, skin_react=k)
            for a in (lo, hi) for b in (15, 45) for o in (0, 480) for s in (0, 1) for k in (1, 5)]
    p = m["model"].predict(pd.DataFrame(rows)[m["features"]])
    check(key + ": no NaN, all 20-150 nmol/L at extremes", np.isfinite(p).all() and p.min() > 20 and p.max() < 150,
          "min %.0f max %.0f" % (p.min(), p.max()))

# ---------------------------------------------------------------- 5. full server pipeline
print("\n=== 5. server.py predict_for() on test profiles ===")
try:
    import server
    base = {"age": 22, "sex": "female", "bmi": 23.8, "supplement": "no", "outdoor_wd": 60, "outdoor_we": 120,
            "answers": [2, 2, 2, 2, 2], "sunscreen": "never"}
    cases = [("baseline 22F", {}), ("supplement yes", {"supplement": "yes"}), ("BMI 35", {"bmi": 35}),
             ("outdoors 0 min", {"outdoor_wd": 0, "outdoor_we": 0}), ("outdoors 240 min", {"outdoor_wd": 240, "outdoor_we": 240}),
             ("burns badly (answer 0)", {"answers": [2, 0, 2, 2, 2]}), ("never burns (answer 4)", {"answers": [2, 4, 2, 2, 2]}),
             ("age 65", {"age": 65}), ("age 76", {"age": 76}), ("age 16", {"age": 16}), ("age 85", {"age": 85}),
             ("old profile (no outdoor)", {"outdoor_wd": None, "outdoor_we": None})]
    for name, ch in cases:
        r = server.predict_for(dict(base, **ch))
        if r is None:
            print("   %-26s -> no prediction (RDA %d)" % (name, server.rda_goal(ch.get("age", 22))))
        else:
            print("   %-26s -> %-4s level %5.1f  gap %5.1f  gap_iu %4d  goal %4d  %s"
                  % (name, r["model_used"], r["level"], r["gap"], r["gap_iu"], r["goal_iu"], r["cat"]))
    a = server.predict_for(base)["level"]
    check("supplement yes raises level", server.predict_for(dict(base, supplement="yes"))["level"] > a)
    check("higher BMI lowers level", server.predict_for(dict(base, bmi=35))["level"] < a)
    check("age 16 and 85 give no prediction", server.predict_for(dict(base, age=16)) is None and server.predict_for(dict(base, age=85)) is None)
except Exception as e:
    print("[SKIP] server test not run:", repr(e))

print("\n=== Summary: %d of %d checks passed ===" % (sum(results), len(results)))