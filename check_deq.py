import pandas as pd

df = pd.read_sas("data/DEQ_H.XPT", format="xport")
print(df.shape)
print(df.columns.tolist())
cols = [c for c in ["DED120", "DED125", "DEQ034D", "DED031"] if c in df.columns]
print(df[cols].describe().round(1))