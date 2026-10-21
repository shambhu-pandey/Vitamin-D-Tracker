
from pathlib import Path
import pandas as pd

data_folder = Path("data")
data_folder.mkdir(exist_ok=True)

files = list(data_folder.glob("*.XPT")) + list(data_folder.glob("*.xpt"))

for file in files:
    try:
        df = pd.read_sas(file, format="xport")
        output_file = file.with_suffix(".csv")
        df.to_csv(output_file, index=False)

        print(f"SUCCESS: {file.name} -> {output_file.name}")

    except Exception as e:
        print(f"ERROR: {file.name}: {e}")

print("Conversion process completed!")










