"""download_more.py - downloads the extra NHANES 2013-2014 files needed by train_v2.py
Run inside vitd_project:   python download_more.py
"""
import os
import urllib.request

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/"
FILES = ["KIQ_U_H",   # kidney conditions
         "MCQ_H",     # medical conditions (liver, thyroid)
         "DIQ_H",     # diabetes
         "ALQ_H",     # alcohol
         "OSQ_H",     # osteoporosis
         "SMQ_H",     # smoking
         "DBQ_H"]     # diet behaviour (milk)

os.makedirs("data", exist_ok=True)
for name in FILES:
    out = os.path.join("data", name + ".XPT")
    if os.path.exists(out):
        print(name, "already there")
        continue
    done = False
    for ext in (".xpt", ".XPT"):
        try:
            req = urllib.request.Request(BASE + name + ext, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if not data.startswith(b"HEADER RECORD"):
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
        print(">>> could not download", name, "(the script will simply skip the features that need it)")