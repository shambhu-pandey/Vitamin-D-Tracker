"""Leakage-safe evaluation for LBXVIDMS < 30 nmol/L classification.

This script does not alter the existing regression artifacts. Outer folds provide
the reported estimates; thresholds are selected only from inner training-fold
predictions.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict


DATA = "data"
SEED = 42
OUTER = StratifiedKFold(5, shuffle=True, random_state=SEED)
INNER = StratifiedKFold(3, shuffle=True, random_state=SEED + 1)
INNER_REGRESSION = KFold(3, shuffle=True, random_state=SEED + 1)


def read(name):
    return pd.read_sas(os.path.join(DATA, name + ".XPT"), format="xport")


def make_data():
    demo = read("DEMO_H")[["SEQN", "RIDAGEYR", "RIAGENDR"]]
    bmx = read("BMX_H")[["SEQN", "BMXBMI"]]
    vid = read("VID_H")[["SEQN", "LBXVIDMS"]]
    dsq = read("DSQTOT_H")[["SEQN", "DSQTVD"]]
    deq = read("DEQ_H")[["SEQN", "DED120", "DED125", "DED031"]]
    base = demo.merge(bmx, on="SEQN").merge(vid, on="SEQN").merge(
        dsq, on="SEQN", how="left")
    base = base.dropna(subset=["BMXBMI", "LBXVIDMS"])
    base["DSQTVD"] = base["DSQTVD"].fillna(0)
    base = base[base["RIDAGEYR"] >= 18]

    sun = base.merge(deq, on="SEQN")
    wd = sun["DED120"].where(sun["DED120"] <= 480)
    we = sun["DED125"].where(sun["DED125"] <= 480)

    def features(frame, with_sun):
        x = pd.DataFrame({
            "age": frame["RIDAGEYR"],
            "female": (frame["RIAGENDR"] == 2).astype(float),
            "bmi": frame["BMXBMI"],
            "supplement": (frame["DSQTVD"] > 0).astype(float),
        }, index=frame.index)
        if with_sun:
            x["outdoor_min"] = ((5 * wd + 2 * we) / 7).fillna(we).fillna(wd)
            x["skin_react"] = frame["DED031"].where(
                frame["DED031"].isin([1, 2, 3, 4, 5]))
        return x

    output = []
    for name, frame, with_sun, features_used, age_filter in (
            ("base", base, False, ["age", "female", "bmi", "supplement"], None),
            ("sun", sun, True, ["age", "female", "bmi", "supplement",
                                "outdoor_min", "skin_react"], (20, 59))):
        x = features(frame, with_sun)
        keep = x.notna().all(axis=1)
        if age_filter:
            keep &= frame["RIDAGEYR"].between(*age_filter)
        x = x.loc[keep, features_used]
        y_level = frame.loc[keep, "LBXVIDMS"].astype(float)
        output.append((name, x.reset_index(drop=True),
                       y_level.reset_index(drop=True)))
    return output


def scores(y, probability, threshold):
    pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "actual_deficient": int(y.sum()),
        "predicted_deficient": int(pred.sum()),
        "true_positives": int(tp), "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "accuracy": round(float(accuracy_score(y, pred)), 3),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 3),
        "specificity": round(float(tn / max(tn + fp, 1)), 3),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 3),
        "pr_auc": round(float(average_precision_score(y, probability)), 3),
        "roc_auc": round(float(roc_auc_score(y, probability)), 3),
        "threshold": round(float(threshold), 4),
    }


def bootstrap_intervals(y, probability, threshold, iterations=1000):
    """Percentile intervals use only the already-held-out outer predictions."""
    rng = np.random.default_rng(SEED)
    y = np.asarray(y)
    probability = np.asarray(probability)
    values = {name: [] for name in
              ("accuracy", "precision", "recall", "specificity", "f1")}
    for _ in range(iterations):
        indexes = rng.integers(0, len(y), len(y))
        sample = scores(y[indexes], probability[indexes], threshold)
        for name in values:
            values[name].append(sample[name])
    return {
        name: [round(float(np.percentile(sample, 2.5)), 3),
               round(float(np.percentile(sample, 97.5)), 3)]
        for name, sample in values.items()
    }


def choose_threshold(y, probability):
    candidates = np.linspace(0.05, 0.80, 151)
    best = max(candidates, key=lambda t: (
        f1_score(y, probability >= t, zero_division=0),
        recall_score(y, probability >= t, zero_division=0),
        -abs(t - 0.5)))
    return float(best)


def choose_level_threshold(y, predicted_level):
    candidates = np.arange(10.0, 81.0, 0.5)
    return float(max(
        candidates,
        key=lambda t: (
            f1_score(y, predicted_level < t, zero_division=0),
            recall_score(y, predicted_level < t, zero_division=0),
            -abs(t - 30.0))))


def operating_points(y, score, thresholds, lower_is_positive=False):
            points = []
            for threshold in thresholds:
                points.append(scores(
                    y, score, -threshold if lower_is_positive else threshold))
                points[-1]["threshold"] = round(float(threshold), 2)
            return points


def evaluate(name, x, y_level):
    y = (y_level.to_numpy() < 30).astype(int)
    results = {"people": len(y), "prevalence": round(float(y.mean()), 3),
               "models": {}}
    candidates = {
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=4, class_weight="balanced",
            random_state=SEED, n_jobs=-1),
        "hist_gradient_boosting_balanced": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
            class_weight="balanced", random_state=SEED),
    }
    regression = joblib.load(
        "vitd_model_v2.joblib" if name == "base" else "vitd_model_sun_v2.joblib")

    for model_name, estimator in {
            "regression_threshold": regression["model"],
            **candidates}.items():
        outer_prob, outer_true = [], []
        outer_thresholds = []
        for train, test in OUTER.split(x, y):
            if model_name == "regression_threshold":
                inner_pred = cross_val_predict(clone(estimator), x.iloc[train],
                                               y_level.iloc[train],
                                               cv=INNER_REGRESSION)
                threshold = choose_level_threshold(y[train], inner_pred)
                fitted = clone(estimator).fit(x.iloc[train], y_level.iloc[train])
                test_level = fitted.predict(x.iloc[test])
                test_prob = -test_level
                outer_thresholds.append(threshold)
                outer_prob.extend(test_prob)
                outer_true.extend(y[test])
                continue
            else:
                inner_prob = cross_val_predict(
                    clone(estimator), x.iloc[train], y[train], cv=INNER,
                    method="predict_proba")[:, 1]
                threshold = choose_threshold(y[train], inner_prob)
                fitted = clone(estimator).fit(x.iloc[train], y[train])
                test_prob = fitted.predict_proba(x.iloc[test])[:, 1]
            outer_prob.extend(test_prob)
            outer_true.extend(y[test])
            outer_thresholds.append(threshold)
        if model_name == "regression_threshold":
            levels_score = np.asarray(outer_prob)
            y_array = np.asarray(outer_true)
            cut = float(np.mean(outer_thresholds))
            pred = (-levels_score < cut).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_array, pred, labels=[0, 1]).ravel()
            result = {
                "actual_deficient": int(y_array.sum()),
                "predicted_deficient": int(pred.sum()),
                "true_positives": int(tp), "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
                "accuracy": round(float(accuracy_score(y_array, pred)), 3),
                "precision": round(float(precision_score(y_array, pred, zero_division=0)), 3),
                "recall": round(float(recall_score(y_array, pred, zero_division=0)), 3),
                "specificity": round(float(tn / max(tn + fp, 1)), 3),
                "f1": round(float(f1_score(y_array, pred, zero_division=0)), 3),
                "pr_auc": round(float(average_precision_score(y_array, levels_score)), 3),
                "roc_auc": round(float(roc_auc_score(y_array, levels_score)), 3),
                "threshold": round(cut, 2),
            }
            result["confidence_intervals_95"] = bootstrap_intervals(
                y_array, levels_score, -cut, iterations=1000)
            result["operating_points"] = operating_points(
                y_array, levels_score, [30, 40, 50, 54.5, 60, 70],
                lower_is_positive=True)
        else:
            result = scores(np.asarray(outer_true), np.asarray(outer_prob),
                            float(np.mean(outer_thresholds)))
            result["confidence_intervals_95"] = bootstrap_intervals(
                np.asarray(outer_true), np.asarray(outer_prob),
                float(np.mean(outer_thresholds)), iterations=1000)
            result["operating_points"] = operating_points(
                np.asarray(outer_true), np.asarray(outer_prob),
                [0.2, 0.3, 0.4, 0.5, 0.6])
        result["outer_thresholds"] = [round(v, 4) for v in outer_thresholds]
        results["models"][model_name] = result
    return results


def main():
    report = {"threshold_selection": "inner 3-fold OOF within each outer training fold",
              "outer_folds": 5, "threshold_target": "F1 with recall tie-break",
              "models": {}}
    for name, x, y in make_data():
        print("\n" + "=" * 72)
        print(name, "rows:", len(y), "actual deficient:", int((y < 30).sum()))
        result = evaluate(name, x, y)
        report["models"][name] = result
        for model_name, metrics in result["models"].items():
            print(model_name, json.dumps(metrics, sort_keys=True))
    with open("deficiency_validation_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("\nSaved deficiency_validation_report.json")


if __name__ == "__main__":
    main()
