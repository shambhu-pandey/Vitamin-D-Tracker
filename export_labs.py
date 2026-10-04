import csv
import hashlib
import json
import sqlite3

c = sqlite3.connect("tracker.db")
c.row_factory = sqlite3.Row
rows = c.execute("SELECT * FROM lab_results WHERE consent = 1 ORDER BY id").fetchall()

cols = ["person", "taken_on", "measured_nmol", "predicted_nmol", "model_used", "age", "sex", "bmi",
        "supplement", "outdoor_wd", "outdoor_we", "skin_type", "skin_react", "exposed",
        "sunscreen", "quiz_answers"]
with open("lab_dataset.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(cols)
    errs = []
    for r in rows:
        s = json.loads(r["features"])
        person = hashlib.sha256(("salt-" + str(r["user_id"])).encode()).hexdigest()[:8]
        w.writerow([person, r["taken_on"], r["value_nmol"], r["predicted_nmol"], r["model_used"]] +
                   [s.get(k) for k in cols[5:]])
        if r["predicted_nmol"] is not None:
            errs.append(abs(r["predicted_nmol"] - r["value_nmol"]))
print("rows exported:", len(rows))
if errs:
    print("mean absolute error of the app on real results: %.1f nmol/L (n=%d)" % (sum(errs) / len(errs), len(errs)))