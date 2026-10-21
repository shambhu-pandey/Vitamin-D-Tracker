"""Focused tests for the configurable health-adjustment layer."""

from copy import deepcopy

import server


BASE = {
    "age": 35, "sex": "female", "bmi": 24.0, "supplement": "no",
    "outdoor_wd": 60, "outdoor_we": 120, "answers": [2, 2, 2, 2, 2],
    "sunscreen": "sometimes", "kidney": 0, "liver": 0, "thyroid": 0,
    "diabetes": 0, "bone": 0, "smoker": 0, "milk": 0, "alcohol": 0,
}


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    print("[PASS]", name)


none = server.predict_for(BASE)
individual = {
    name: server.predict_for(dict(BASE, **{name: 1}))
    for name in ("kidney", "liver", "thyroid", "diabetes", "bone", "smoker")
}
expected = {
    "kidney": 1.0, "liver": 0.5, "thyroid": 1.5,
    "diabetes": 1.0, "bone": 0.5, "smoker": 1.0,
}
multiple = server.predict_for(dict(BASE, kidney=1, liver=1, thyroid=1,
                                   diabetes=1, bone=1, smoker=1))
all_no = server.predict_for(dict(BASE, kidney=0, liver=0, thyroid=0,
                                  diabetes=0, bone=0, smoker=0))
overlap = server.predict_for(dict(BASE, kidney=1, diabetes=1))
switched = server.predict_for(dict(BASE, kidney=0))

check("no conditions have zero health adjustment",
      none["health_adjustment"] == 0 and none["adjustment_details"] == [])
for name, result in individual.items():
    check("%s individual contribution matches configured deduction" % name,
          result["total_deduction"] == expected[name]
          and result["level"] == result["original_level"] - expected[name])
check("multiple conditions are capped",
      multiple["adjustment_summary"]["binary_applied_total"] == 4.0
      and multiple["adjustment_summary"]["cap_applied"])
configured = deepcopy(server.HEALTH_ADJUSTMENTS)
server.HEALTH_ADJUSTMENTS["binary"] = {
    name: {"adjustment": 2.0}
    for name in ("kidney", "liver", "thyroid", "diabetes", "bone", "smoker")
}
_, _, _, _, capped_summary = server.health_adjustment_for(
    dict(BASE, kidney=1, liver=1, thyroid=1, diabetes=1, bone=1, smoker=1))
server.HEALTH_ADJUSTMENTS = configured
check("configured cap limits a multi-condition pool",
      capped_summary["binary_applied_total"] == 4.0
      and capped_summary["cap_applied"])
check("all conditions remain capped",
      multiple["total_deduction"] <= 4.0)
check("NO removes every binary penalty",
      all_no["health_adjustment"] == 0)
check("switching YES back to NO removes the deduction",
      switched["health_adjustment"] == 0
      and individual["kidney"]["health_adjustment"] == -1.0)
check("kidney and diabetes do not stack their overlap",
      any(a.get("overlap_suppressed") == "kidney disease"
          for a in overlap["adjustment_details"]))
check("original and final predictions are separate",
      multiple["original_level"] != multiple["level"]
      and "adjustment_details" in multiple)
check("binary details identify experimental overrides",
      all(a["experimental_override"] and not a["validation_supported"]
          for a in individual["kidney"]["adjustment_details"]
          if a["input"] == "kidney"))
check("final prediction respects configured bounds",
      none["prediction_bounds"][0] <= multiple["level"]
      <= none["prediction_bounds"][1])
missing = server.predict_for(dict(BASE, kidney=None, liver="unexpected"))
check("missing or unrecognized inputs add no binary deduction",
      missing["health_adjustment"] == 0)
milk_bonus = server.predict_for(dict(BASE, milk=3))
rarely_milk = server.predict_for(dict(BASE, milk=1))
sometimes_milk = server.predict_for(dict(BASE, milk=2))
never_milk = server.predict_for(dict(BASE, milk=0))
unknown_milk = server.predict_for(dict(BASE, milk="unknown"))
check("rarely milk applies the experimental +1.0 bonus",
      rarely_milk["milk_bonus"] == 1.0)
check("sometimes milk applies the experimental +1.5 bonus",
      sometimes_milk["milk_bonus"] == 1.5)
check("every-day milk applies the experimental +2.5 bonus",
      milk_bonus["milk_bonus"] == 2.5)
check("never milk applies no bonus", never_milk["milk_bonus"] == 0)
check("unknown milk input applies no bonus",
      unknown_milk["milk_bonus"] == 0)
check("milk bonus is not part of the binary deduction cap",
      milk_bonus["total_deduction"] == 0
      and milk_bonus["level"] == milk_bonus["original_level"] + 2.5)

print("All focused health-adjustment checks passed.")
