import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

rd = lambda f: pd.read_sas("data/%s.XPT" % f, format="xport")

demo = rd("DEMO_H")[["SEQN", "RIDAGEYR", "RIAGENDR"]]
bmx = rd("BMX_H")[["SEQN", "BMXBMI"]]
vid = rd("VID_H")[["SEQN", "LBXVIDMS"]]
dsq = rd("DSQTOT_H")[["SEQN", "DSQTVD"]]
deq = rd("DEQ_H")[["SEQN", "DED120", "DED125", "DEQ034D", "DED031"]]

df = demo.merge(bmx, on="SEQN").merge(vid, on="SEQN").merge(deq, on="SEQN")
df = df.merge(dsq, on="SEQN", how="left")
df = df.dropna(subset=["BMXBMI", "LBXVIDMS"])
df["DSQTVD"] = df["DSQTVD"].fillna(0)
print("people with dermatology + vitamin D data:", len(df))
print("  does not work/go to school (DED120 = 3333):", int((df["DED120"] == 3333).sum()))
print("  at work 9-5 seven days (DED125 = 3333):", int((df["DED125"] == 3333).sum()))

# minutes outdoors 9am-5pm: 3333 / 7777 / 9999 are codes, not minutes
wd = df["DED120"].where(df["DED120"] <= 480)
we = df["DED125"].where(df["DED125"] <= 480)
outdoor = ((5 * wd + 2 * we) / 7).fillna(we).fillna(wd)     # no workdays -> use non-work days, and so on
sunscreen = df["DEQ034D"].where(df["DEQ034D"].isin([1, 2, 3, 4, 5]))   # 1 always ... 5 never
skin = df["DED031"].where(df["DED031"].isin([1, 2, 3, 4, 5]))          # 1 burns badly ... 5 nothing happens

X_all = pd.DataFrame({
    "age": df["RIDAGEYR"],
    "female": (df["RIAGENDR"] == 2).astype(float),
    "bmi": df["BMXBMI"],
    "supplement": (df["DSQTVD"] > 0).astype(float),
    "outdoor_min": outdoor,
    "sunscreen": sunscreen,
    "skin_react": skin,
})
keep = X_all.notna().all(axis=1)
X_all, y = X_all[keep], df.loc[keep, "LBXVIDMS"]
print("rows used:", len(X_all), "| mean level: %.1f | std: %.1f" % (y.mean(), y.std()))
print("baseline (always the average): MAE = %.1f" % mean_absolute_error(y, [y.mean()] * len(y)))

BASE = ["age", "female", "bmi", "supplement"]
SETS = {
    "base (4)": BASE,
    "+ outdoor_min": BASE + ["outdoor_min"],
    "+ skin_react": BASE + ["skin_react"],
    "+ sunscreen": BASE + ["sunscreen"],
    "+ outdoor_min + skin_react": BASE + ["outdoor_min", "skin_react"],
    "+ all 3 sun features": BASE + ["outdoor_min", "sunscreen", "skin_react"],
}


def make(name):
    if name == "linear":
        return LinearRegression()
    return GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                     subsample=0.8, random_state=42)


def cv_scores(mname, X):
    m, r, q = [], [], []
    for seed in (1, 2, 3):                       # 3 different 5-fold splits, averaged
        pred = cross_val_predict(make(mname), X, y, cv=KFold(5, shuffle=True, random_state=seed))
        m.append(mean_absolute_error(y, pred))
        r.append(mean_squared_error(y, pred) ** 0.5)
        q.append(r2_score(y, pred))
    return float(np.mean(m)), float(np.mean(r)), float(np.mean(q))


rows = []
print("\ncross-validation (same people in every row):")
for sname, cols in SETS.items():
    for mname in ("linear", "gradient_boosting"):
        mae, rmse, r2 = cv_scores(mname, X_all[cols])
        rows.append((sname, mname, cols, mae, rmse, r2))
        print("  %-28s %-18s MAE = %.2f  RMSE = %.1f  R2 = %.3f" % (sname, mname, mae, rmse, r2))

# pick: best MAE, but prefer fewer features if it is within 0.2 of the best
best_mae = min(r[3] for r in rows)
ok = [r for r in rows if r[3] <= best_mae + 0.2]
sname, mname, cols, mae, rmse, r2 = min(ok, key=lambda r: (len(r[2]), r[3]))
print("\nchosen (simplest within 0.2 of the best): %s + %s | MAE %.2f R2 %.3f" % (sname, mname, mae, r2))

lin = LinearRegression().fit(X_all[cols], y)
print("linear effects on the chosen features (nmol/L per unit):")
for f, c in zip(cols, lin.coef_):
    print("  %-12s %+.3f" % (f, c))

model = make(mname).fit(X_all[cols], y)
if "outdoor_min" in cols:
    print("\nexample (22 y, female, BMI 23.8, no supplement, sunscreen never, skin reaction 3):")
    for mins in (0, 30, 60, 120, 240):
        row = {"age": 22, "female": 1.0, "bmi": 23.8, "supplement": 0.0,
               "outdoor_min": mins, "sunscreen": 5, "skin_react": 3}
        print("  %3d min outdoors/day -> %.1f nmol/L" %
              (mins, model.predict(pd.DataFrame([row])[cols])[0]))

joblib.dump({"kind": "regression", "model": model, "features": cols, "name": mname,
             "mae": mae, "rmse": rmse, "r2": r2, "rows": int(len(X_all)),
             "age_range": [20, 59]}, "vitd_model_sun.joblib")
print("\nsaved vitd_model_sun.joblib")