"""Train core models and derive the separate NHANES health-adjustment layer.

The Gradient Boosting models intentionally exclude the health/lifestyle fields
handled here.  Adjustments are contrasts of out-of-fold residual means after
the core model has accounted for age, sex, BMI, supplementation and (for the
sun model) outdoor exposure and skin reaction.  They are project heuristics,
not clinically validated effects.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


DATA = "data"
MODEL_ARGS = dict(n_estimators=150, max_depth=2, learning_rate=0.05,
                  subsample=0.8, random_state=42)
MAX_SINGLE_HEALTH_DEDUCTION_NMOL = 2.0
MAX_HEALTH_DEDUCTION_NMOL = 2.0
DEMO_BINARY_ADJUSTMENTS_NMOL = {
    "kidney": 1.0, "liver": 0.5, "thyroid": 1.5,
    "diabetes": 1.0, "bone": 0.5, "smoker": 1.0,
}
BASE_FEATURES = ["age", "female", "bmi", "supplement"]
SUN_FEATURES = BASE_FEATURES + ["outdoor_min", "skin_react"]


def read(name):
    return pd.read_sas(os.path.join(DATA, name + ".XPT"), format="xport")


def yes_no(series, yes=(1,), no=(2,)):
    return series.map(lambda value: 1.0 if value in yes
                      else (0.0 if value in no else np.nan))


def build_frame():
    df = (read("DEMO_H")[["SEQN", "RIDAGEYR", "RIAGENDR"]]
          .merge(read("BMX_H")[["SEQN", "BMXBMI"]], on="SEQN")
          .merge(read("VID_H")[["SEQN", "LBXVIDMS"]], on="SEQN")
          .merge(read("DSQTOT_H")[["SEQN", "DSQTVD"]], on="SEQN", how="left"))
    for name, columns in {
        "DEQ_H": ["DED120", "DED125", "DED031"],
        "KIQ_U_H": ["KIQ022"],
        "MCQ_H": ["MCQ160L", "MCQ160M"],
        "DIQ_H": ["DIQ010"],
        "OSQ_H": ["OSQ060"],
        "SMQ_H": ["SMQ020", "SMQ040"],
        "DBQ_H": ["DBQ197"],
        "ALQ_H": ["ALQ101", "ALQ120Q", "ALQ120U", "ALQ130"],
    }.items():
        table = read(name)
        df = df.merge(table[["SEQN"] + columns], on="SEQN", how="left")
    df = df[(df["RIDAGEYR"] >= 18) & df["BMXBMI"].notna()
            & df["LBXVIDMS"].notna()].copy()
    df["DSQTVD"] = df["DSQTVD"].fillna(0)
    return df.reset_index(drop=True)


def health_columns(df):
    out = pd.DataFrame(index=df.index)
    out["kidney"] = yes_no(df["KIQ022"])
    out["liver"] = yes_no(df["MCQ160L"])
    out["thyroid"] = yes_no(df["MCQ160M"])
    out["diabetes"] = yes_no(df["DIQ010"], yes=(1,), no=(2, 3))
    out["bone"] = yes_no(df["OSQ060"])
    smoker = pd.Series(np.nan, index=df.index)
    smoker[df["SMQ020"] == 2] = 0.0
    smoker[df["SMQ040"].isin([1, 2])] = 1.0
    smoker[df["SMQ040"] == 3] = 0.0
    out["smoker"] = smoker

    q = df["ALQ120Q"].where(df["ALQ120Q"] <= 366)
    per_week = q.where(df["ALQ120U"] == 1)
    per_week = per_week.fillna((q / 4.345).where(df["ALQ120U"] == 2))
    per_week = per_week.fillna((q / 52.14).where(df["ALQ120U"] == 3))
    alcohol = (per_week * df["ALQ130"].where(df["ALQ130"] <= 60))
    alcohol[q == 0] = 0.0
    alcohol[(df["ALQ101"] == 2) & alcohol.isna()] = 0.0
    out["alcohol_week"] = alcohol.clip(upper=60)
    out["milk"] = df["DBQ197"].where(df["DBQ197"].isin([0, 1, 2, 3]))
    return out


def core_frame(df):
    x = pd.DataFrame({
        "age": df["RIDAGEYR"].astype(float),
        "female": (df["RIAGENDR"] == 2).astype(float),
        "bmi": df["BMXBMI"].astype(float),
        "supplement": (df["DSQTVD"] > 0).astype(float),
    })
    wd = df["DED120"].where(df["DED120"] <= 480)
    we = df["DED125"].where(df["DED125"] <= 480)
    x["outdoor_min"] = ((5 * wd + 2 * we) / 7).fillna(we).fillna(wd)
    x["skin_react"] = df["DED031"].where(df["DED031"].isin([1, 2, 3, 4, 5]))
    return x


def contrast(values, residuals, reference=0.0):
    valid = pd.DataFrame({"value": values, "residual": residuals}).dropna()
    groups = valid.groupby("value", observed=True)["residual"]
    counts = groups.size()
    means = groups.mean()
    if reference not in means.index:
        return None
    return valid, counts, means, float(means.get(reference))


def derive_adjustments(h, residuals):
    result = {
        "description": "Out-of-fold residual contrasts after the core Gradient Boosting model.",
        "unit": "nmol/L",
        "target_valid_range_nmol": [9.37, 318.0],
        "MAX_SINGLE_HEALTH_DEDUCTION_NMOL": MAX_SINGLE_HEALTH_DEDUCTION_NMOL,
        "MAX_HEALTH_DEDUCTION_NMOL": MAX_HEALTH_DEDUCTION_NMOL,
        "experimental_milk_bonus_nmol": 0.0,
        "experimental_milk_bonus_by_condition": {
            "never": 0.0, "rarely": 1.0, "sometimes": 1.5, "every_day": 2.5,
        },
        "milk_bonus_status": "Experimental academic demo assumptions; not scientifically validated.",
        "cap_method": "Positive binary deductions are limited to 2.0 nmol/L individually and combined after the kidney-diabetes overlap rule. Raw estimates are preserved.",
        "cap_status": "Experimental UI heuristics; not medically validated thresholds.",
        "residual_method": "Five-fold out-of-fold residuals from core Gradient Boosting, random_state 42. Negative estimates are shrunk by max(0, abs(raw contrast) - standard error); non-negative observed associations receive zero binary penalty.",
        "binary": {},
        "alcohol": {},
        "milk": {},
    }
    for name in ("kidney", "liver", "thyroid", "diabetes", "bone", "smoker"):
        c = contrast(h[name], residuals)
        if c is None:
            result["binary"][name] = {"adjustment": 0.0, "reliable": False,
                                      "reason": "No training-set reference group."}
            continue
        valid, counts, means, no_mean = c
        yes_mean = float(means.get(1.0, np.nan))
        raw = yes_mean - no_mean
        yes_group = valid.loc[valid["value"] == 1.0, "residual"]
        no_group = valid.loc[valid["value"] == 0.0, "residual"]
        standard_error = float(np.sqrt(
            yes_group.var(ddof=1) / max(len(yes_group), 1)
            + no_group.var(ddof=1) / max(len(no_group), 1)))
        # Retain only the negative amount beyond one standard error. A
        # positive observed contrast is not converted into a negative claim.
        shrunk = max(0.0, abs(raw) - standard_error) if raw < 0 else 0.0
        evidence_deduction = min(shrunk, MAX_SINGLE_HEALTH_DEDUCTION_NMOL)
        result["binary"][name] = {
            "adjustment": DEMO_BINARY_ADJUSTMENTS_NMOL[name],
            "evidence_supported_deduction": round(evidence_deduction, 3),
            "experimental_override": True,
            "validation_supported": False,
            "applied_method": "Fixed provisional YES deduction; not the raw NHANES contrast.",
            "raw_residual_contrast": round(raw, 3),
            "standard_error": round(standard_error, 3),
            "shrinkage_rule": "negative effect beyond one standard error; otherwise zero, stored as a positive deduction magnitude",
            "yes_mean_residual": round(yes_mean, 3),
            "no_mean_residual": round(no_mean, 3),
            "n_valid": int(len(valid)),
            "n_yes": int(counts.get(1.0, 0)),
            "n_no": int(counts.get(0.0, 0)),
            "missing": int(h[name].isna().sum()),
            "reliable": bool(raw < 0 and abs(raw) > standard_error
                             and counts.get(1.0, 0) >= 100),
            "reason": ("Observed held-out association was non-negative; no "
                       "negative binary penalty applied."
                       if raw >= 0 else "Negative observational contrast shrunk by its standard error."),
        }

    alcohol_week = h["alcohol_week"].map(
        lambda value: np.nan if pd.isna(value) else
        (0 if value == 0 else 1 if value <= 7 else 2 if value <= 14 else 3))
    c = contrast(alcohol_week, residuals)
    if c:
        valid, counts, means, none_mean = c
        for category, label in enumerate(("none", "1_to_7", "8_to_14", "15_plus")):
            raw = float(means.get(category, none_mean) - none_mean)
            result["alcohol"][label] = {
                "adjustment": round(raw, 3), "raw_residual_contrast": round(raw, 3),
                "n_valid": int(len(valid)), "n_category": int(counts.get(category, 0)),
                "missing": int(h["alcohol_week"].isna().sum()),
                "reliable": bool(category == 0 or counts.get(category, 0) >= 100),
                "reason": "Compared with no alcohol; observational and not causal.",
            }

    milk_valid = pd.DataFrame({"value": h["milk"], "residual": residuals}).dropna()
    if not milk_valid.empty:
        groups = milk_valid.groupby("value", observed=True)["residual"]
        counts, means = groups.size(), groups.mean()
        reference_mean = float(milk_valid["residual"].mean())
        # Mixed category contrasts are not a defensible monotonic effect.
        for category, label in enumerate(("never", "rarely", "sometimes", "every_day")):
            result["milk"][label] = {
                "adjustment": 0.0, "observed_residual_contrast": round(
                    float(means.get(float(category), reference_mean) - reference_mean), 3),
                "n_valid": int(len(milk_valid)), "n_category": int(counts.get(float(category), 0)),
                "missing": int(h["milk"].isna().sum()), "reliable": False,
                "reason": "No adjustment: category contrasts are not consistently directional.",
            }
    return result


def train_model(x, y, features, output, name, age_range=None):
    train, test = train_test_split(np.arange(len(x)), test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(**MODEL_ARGS).fit(x.iloc[train][features], y.iloc[train])
    pred = model.predict(x.iloc[test][features])
    artifact = {"kind": "regression", "model": model, "features": features,
                "name": name, "mae": float(mean_absolute_error(y.iloc[test], pred)),
                "r2": float(r2_score(y.iloc[test], pred)), "rows": int(len(x))}
    if age_range:
        artifact["age_range"] = list(age_range)
    joblib.dump(artifact, output)
    return train, test, artifact


def main():
    df = build_frame()
    x = core_frame(df)
    y = df["LBXVIDMS"].astype(float)
    h = health_columns(df)
    train, _, base = train_model(x, y, BASE_FEATURES, "vitd_model_v2.joblib",
                                  "gradient_boosting_core")
    sun = x[SUN_FEATURES].notna().all(axis=1) & df["RIDAGEYR"].between(20, 59)
    sun_train, _, sun_artifact = train_model(
        x.loc[sun].reset_index(drop=True), y.loc[sun].reset_index(drop=True),
        SUN_FEATURES, "vitd_model_sun_v2.joblib", "gradient_boosting_core_sun",
        (20, 59))
    # Derive health effects only from out-of-fold predictions so each residual
    # comes from a model that did not train on that participant.
    oof = np.zeros(len(x))
    for fold_train, fold_test in KFold(5, shuffle=True, random_state=42).split(x):
        fold_model = GradientBoostingRegressor(**MODEL_ARGS).fit(
            x.iloc[fold_train][BASE_FEATURES], y.iloc[fold_train])
        oof[fold_test] = fold_model.predict(x.iloc[fold_test][BASE_FEATURES])
    adjustments = derive_adjustments(h, y.to_numpy() - oof)
    adjustments["rows_total"] = int(len(df))
    adjustments["rows_core_training"] = int(len(train))
    adjustments["rows_sun_training"] = int(len(sun_train))
    with open("health_adjustments.json", "w", encoding="utf-8") as handle:
        json.dump(adjustments, handle, indent=2)
    print(json.dumps(adjustments, indent=2))


if __name__ == "__main__":
    main()
