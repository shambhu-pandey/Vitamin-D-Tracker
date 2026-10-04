import pandas as pd

lines = []
for f in ["DEMO_H", "BMX_H", "VID_H", "DSQTOT_H"]:
    reader = pd.read_sas("data/%s.XPT" % f, format="xport", iterator=True)
    labels = {}
    try:
        for fld in reader.fields:
            labels[fld["name"]] = fld.get("label", "")
    except Exception:
        pass
    df = reader.read()
    lines.append("\n===== %s  (%d rows, %d columns) =====" % (f, len(df), df.shape[1]))
    for c in df.columns:
        lines.append("%-10s %-60s filled: %d" % (c, str(labels.get(c, ""))[:60], df[c].notna().sum()))

text = "\n".join(lines)
print(text)
with open("columns.txt", "w", encoding="utf-8") as fh:
    fh.write(text)
print("\nsaved columns.txt")