import json
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

rd = lambda f: pd.read_sas("data/%s.XPT" % f, format="xport")

demo = rd("DEMO_H")[["SEQN", "RIDAGEYR", "RIAGENDR"]]
bmx = rd("BMX_H")[["SEQN", "BMXBMI"]]
vid = rd("VID_H")[["SEQN", "LBXVIDMS"]]
dsq = rd("DSQTOT_H")[["SEQN", "DSQTVD"]]

df = demo.merge(bmx, on="SEQN").merge(vid, on="SEQN")
df = df.merge(dsq, on="SEQN", how="left")          # not in the supplement file = no supplement
df = df[df["RIDAGEYR"] >= 18]                       # adults only
df = df.dropna(subset=["BMXBMI", "LBXVIDMS"])
df["DSQTVD"] = df["DSQTVD"].fillna(0)

# No season feature: NHANES season is a US season, not valid for India.
X = pd.DataFrame({
    "age": df["RIDAGEYR"],
    "female": (df["RIAGENDR"] == 2).astype(float),
    "bmi": df["BMXBMI"],
    "supplement": (df["DSQTVD"] > 0).astype(float),
})
y = df["LBXVIDMS"]                                  # serum 25(OH)D in nmol/L
print("rows:", len(X), "| mean level: %.1f nmol/L | std: %.1f" % (y.mean(), y.std()))

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

base_mae = mean_absolute_error(yte, [ytr.mean()] * len(yte))
print("baseline (always predict the average): MAE = %.1f nmol/L" % base_mae)

models = {
    "linear": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=30, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=150, max_depth=2,
                                                   learning_rate=0.05, subsample=0.8,
                                                   random_state=42),
}
best, best_mae, best_name, report = None, 1e9, None, {"models": {}}
for name, m in models.items():
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    mae = mean_absolute_error(yte, pred)
    rmse = mean_squared_error(yte, pred) ** 0.5
    r2 = r2_score(yte, pred)
    print("\n== %-18s MAE = %.1f  RMSE = %.1f  R2 = %.3f" % (name, mae, rmse, r2))
    report["models"][name] = {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 3)}
    if mae < best_mae:
        best, best_mae, best_name, best_rmse, best_r2 = m, mae, name, rmse, r2

lin = models["linear"]
print("\nlinear model, effect on level (nmol/L):",
      {f: round(float(c), 2) for f, c in zip(X.columns, lin.coef_)})

example = pd.DataFrame([{"age": 22, "female": 1.0, "bmi": 23.8, "supplement": 0.0}])
print("example (22 y, female, BMI 23.8, no supplement): %.1f nmol/L" % best.predict(example)[0])

report.update({"rows": int(len(X)), "baseline_mae": round(base_mae, 2), "best": best_name,
               "linear_effects": {f: round(float(c), 3) for f, c in zip(X.columns, lin.coef_)}})
joblib.dump({"kind": "regression", "model": best, "features": list(X.columns),
             "name": best_name, "mae": float(best_mae), "rmse": float(best_rmse),
             "r2": float(best_r2), "rows": int(len(X))}, "vitd_model.joblib")
with open("model_report.json", "w") as f:
    json.dump(report, f, indent=2)
print("\nsaved vitd_model.joblib using", best_name)