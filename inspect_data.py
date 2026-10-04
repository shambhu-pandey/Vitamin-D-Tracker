import pandas as pd

wanted = {"DEMO_H": ["RIDAGEYR", "RIAGENDR", "RIDEXMON"],
          "BMX_H": ["BMXBMI"],
          "VID_H": ["LBXVIDMS"],
          "DSQTOT_H": ["DSQTVD"]}

for f, cols in wanted.items():
    df = pd.read_sas("data/%s.XPT" % f, format="xport")
    print("\n==", f, df.shape)
    for col in cols:
        print(col, "-> found" if col in df.columns else "-> MISSING")
    print(df[[c for c in cols if c in df.columns]].describe().round(1))