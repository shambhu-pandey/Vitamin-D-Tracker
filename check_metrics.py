"""check_metrics.py - accuracy, precision, recall, F1, AUC for both models (honest, cross-validated)
Run inside vitd_project:  python check_metrics.py
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, precision_score, r2_score, recall_score,
                             roc_auc_score, classification_report)
from sklearn.model_selection import KFold, cross_val_predict

rd = lambda f: pd.read_sas("data/%s.XPT" % f, format="xport")

demo = rd("DEMO_H")[["SEQN", "RIDAGEYR", "RIAGENDR"]]
bmx = rd("BMX_H")[["SEQN", "BMXBMI"]]
vid = rd("VID_H")[["SEQN", "LBXVIDMS"]]
dsq = rd("DSQTOT_H")[["SEQN", "DSQTVD"]]
deq = rd("DEQ_H")[["SEQN", "DED120", "DED125", "DEQ034D", "DED031"]]

# same data preparation as training
base_df = demo.merge(bmx, on="SEQN").merge(vid, on="SEQN").merge(dsq, on="SEQN", how="left")
base_df = base_df.dropna(subset=["BMXBMI", "LBXVIDMS"])
base_df["DSQTVD"] = base_df["DSQTVD"].fillna(0)
base_df = base_df[base_df["RIDAGEYR"] >= 18]

sun_df = base_df.merge(deq, on="SEQN")
wd = sun_df["DED120"].where(sun_df["DED120"] <= 480)
we = sun_df["DED125"].where(sun_df["DED125"] <= 480)


def features(d, with_sun):
    X = pd.DataFrame({"age": d["RIDAGEYR"], "female": (d["RIAGENDR"] == 2).astype(float),
                      "bmi": d["BMXBMI"], "supplement": (d["DSQTVD"] > 0).astype(float)})
    if with_sun:
        X["outdoor_min"] = ((5 * wd + 2 * we) / 7).fillna(we).fillna(wd)
        X["skin_react"] = d["DED031"].where(d["DED031"].isin([1, 2, 3, 4, 5]))
    return X


def binary_report(y_true_level, pred_level, cut, label):
    """positive class = true level < cut. Predict positive when predicted level < cut."""
    yt = (y_true_level < cut).astype(int)
    yp = (pred_level < cut).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
    print("\n  %s  (cut-off %d nmol/L)  | really below cut-off: %.1f%% of people" % (label, cut, 100 * yt.mean()))
    print("    accuracy          %.3f" % accuracy_score(yt, yp))
    print("    precision         %.3f   (of people flagged low, how many really are low)" % precision_score(yt, yp, zero_division=0))
    print("    recall            %.3f   (of really-low people, how many we caught)" % recall_score(yt, yp))
    print("    specificity       %.3f   (of not-low people, how many we left alone)" % (tn / (tn + fp)))
    print("    F1                %.3f" % f1_score(yt, yp))
    print("    balanced accuracy %.3f" % balanced_accuracy_score(yt, yp))
    print("    ROC AUC           %.3f   (0.5 = guess, 1.0 = perfect; uses the level itself, any cut-off)"
          % roc_auc_score(yt, -pred_level))
    print("    confusion matrix: TN=%d FP=%d FN=%d TP=%d" % (tn, fp, fn, tp))
    # best cut on the predicted level for F1 (models shrink towards the average, so 50 is not always best)
    best = max(range(40, 90), key=lambda t: f1_score(yt, (pred_level < t).astype(int)))
    yb = (pred_level < best).astype(int)
    print("    -> best F1 if we flag 'predicted < %d': precision %.3f recall %.3f F1 %.3f"
          % (best, precision_score(yt, yb, zero_division=0), recall_score(yt, yb), f1_score(yt, yb)))


def three_class(level):
    return np.where(level < 50, 0, np.where(level < 75, 1, 2))


for key, fn, d, with_sun in (("BASE model (age 18+, 4 features)", "vitd_model.joblib", base_df, False),
                             ("SUN model (age 20-59, 6 features)", "vitd_model_sun.joblib", sun_df, True)):
    m = joblib.load(fn)
    X = features(d, with_sun)
    y = d["LBXVIDMS"]
    keep = X.notna().all(axis=1)
    if with_sun:
        keep &= d["RIDAGEYR"].between(20, 59)
    X, y = X[keep][m["features"]], y[keep]

    # honest predictions: every person is predicted by a model that never saw them
    preds = []
    for seed in (1, 2, 3):
        preds.append(cross_val_predict(clone(m["model"]), X, y, cv=KFold(5, shuffle=True, random_state=seed)))
    pred = np.mean(preds, axis=0)

    print("\n" + "=" * 78)
    print(key, "| people:", len(y))
    print("=" * 78)
    print("REGRESSION (level in nmol/L):  MAE %.2f | R2 %.3f | within +-10 nmol/L: %.0f%% | within +-20: %.0f%%"
          % (mean_absolute_error(y, pred), r2_score(y, pred),
             100 * np.mean(np.abs(y - pred) <= 10), 100 * np.mean(np.abs(y - pred) <= 20)))

    print("\nYES/NO questions:")
    binary_report(y, pred, 30, "Deficient?")
    binary_report(y, pred, 50, "Low (dashboard 'LOW')?")
    binary_report(y, pred, 75, "Not sufficient (below 75 target)?")

    print("\n3 CATEGORIES exactly like the dashboard (LOW <50 | BORDERLINE 50-75 | GOOD 75+):")
    yt3, yp3 = three_class(y.values), three_class(pred)
    print("    accuracy %.3f | balanced accuracy %.3f | macro F1 %.3f"
          % (accuracy_score(yt3, yp3), balanced_accuracy_score(yt3, yp3), f1_score(yt3, yp3, average="macro")))
    print("    confusion matrix (rows = real, columns = predicted; order LOW, BORDERLINE, GOOD):")
    print(pd.DataFrame(confusion_matrix(yt3, yp3), index=["real LOW", "real BORDER", "real GOOD"],
                       columns=["pred LOW", "pred BORDER", "pred GOOD"]).to_string())
    print(classification_report(yt3, yp3, target_names=["LOW", "BORDERLINE", "GOOD"], zero_division=0, digits=3))
    major = max(np.bincount(yt3)) / len(yt3)
    print("    (baseline: always saying the most common category = %.3f accuracy)" % major)