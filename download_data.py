import os
import urllib.request

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/"
FILES = ["DEMO_H", "BMX_H", "VID_H", "DSQTOT_H", "DEQ_H"]

os.makedirs("data", exist_ok=True)

for name in FILES:
    out = os.path.join("data", name + ".XPT")
    done = False
    for ext in (".xpt", ".XPT"):
        url = BASE + name + ext
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if not data.startswith(b"HEADER RECORD"):      # not a real XPT file
                print(name, ext, "-> not an XPT file, trying next")
                continue
            with open(out, "wb") as f:
                f.write(data)
            print(name, "OK,", round(len(data) / 1024), "KB ->", out)
            done = True
            break
        except Exception as e:
            print(name, ext, "failed:", e)
    if not done:
        print(">>> could not download", name)