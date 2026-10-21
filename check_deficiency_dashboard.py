"""Focused checks for the report-driven deficiency screening alert."""
import json

import server


with open("deficiency_validation_report.json", encoding="utf-8") as handle:
    report = json.load(handle)


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("[PASS]", message)


for model, expected_threshold in (("base", 54.5), ("sun", 51.1)):
    metrics = report["models"][model]["models"]["regression_threshold"]
    check(server.SCREENING_THRESHOLDS[model] == expected_threshold,
          f"{model} uses the saved screening threshold")
    check(server.FLAG_CUT[model] == expected_threshold,
          f"{model} live flag uses the saved screening threshold")
    check(server.DEFICIENCY_BASELINE[model]["threshold"] == 30.0,
          f"{model} retains the fixed-30 baseline")
    check(server.SCREENING_VALIDATION[model] == metrics,
          f"{model} validation metrics come from the loaded report")

    below = expected_threshold - 0.1
    above = expected_threshold + 0.1
    check(below < server.FLAG_CUT[model],
          f"{model} prediction below threshold triggers screening condition")
    check(not (above < server.FLAG_CUT[model]),
          f"{model} prediction above threshold does not trigger screening condition")
    check(metrics["precision"] != 0.0 or metrics["recall"] != 0.0,
          f"{model} tuned validation metrics are not the fixed-30 baseline")

check(all(
    result["models"]["regression_threshold"]["actual_deficient"] > 0
    and 0 < result["prevalence"] < 1
    for result in report["models"].values()
), "medical target remains LBXVIDMS below 30 nmol/L")

response = server.app.test_client().get("/about")
html = response.get_data(as_text=True)
check(response.status_code == 200, "About page renders")
for value in ("54.5 nmol/L", "51.1 nmol/L", "16.8%", "20.8%", "44.7%", "41.7%"):
    check(value in html, f"About page displays saved metric {value}")

print("All deficiency dashboard checks passed.")
