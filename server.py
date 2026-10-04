






"""
Vitamin D Tracker - backend, Phase 6: personal sun table, blood-test log, about page

Run:   python server.py
Open:  http://localhost:5000
"""
import json
import os
import re
import secrets
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timedelta
from functools import lru_cache, wraps

from flask import (Flask, flash, g, jsonify, redirect, render_template,
                   request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tracker.db")
KEY_FILE = os.path.join(BASE, "secret.key")


def load_secret():
    if not os.path.exists(KEY_FILE):
        with open(KEY_FILE, "w") as f:
            f.write(secrets.token_hex(32))
    with open(KEY_FILE) as f:
        return f.read().strip()


app.secret_key = load_secret()
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# ---------------------------------------------------------------- ML models (NHANES regression)
TARGET_NMOL = 75        # target serum 25(OH)D (nmol/L): >= 75 is usually called sufficient
NMOL_PER_100IU = 2.0    # literature: ~2 nmol/L rise in 25(OH)D per 100 IU/day (meta-regression)
MAX_GOAL_IU = 2000      # safety cap for the daily sun goal
MODEL_MIN_AGE, MODEL_MAX_AGE = 18, 80     # NHANES adults (80 means "80 or older")

# "Higher risk" flag: flag people whose predicted level is below this. Cut-offs picked from the
# cross-validated results (best F1 for "really below 50 nmol/L"). FLAG_STATS = (precision %, recall %).
FLAG_CUT = {"base": 64, "sun": 62}
FLAG_STATS = {"base": (44, 82), "sun": (48, 80)}

# Numbers shown in the "How accurate is our model?" section (from check_metrics.py, 5-fold cross-validation).
# If you retrain the models, update these values.
MODEL_REPORT = {"models": {
    "base": {
        "title": "Basic model", "people": 5596, "inputs": "age, sex, BMI, supplement (ages 18-80)",
        "mae": 18.6, "r2": 0.243, "w10": 34, "w20": 63,
        "cuts": [
            {"q": "Below 30 nmol/L (deficient)", "prev": 8.1, "acc": 91.9, "prec": 0.0, "rec": 0.0, "spec": 100.0, "f1": 0.0, "auc": 0.718},
            {"q": "Below 50 nmol/L (low)", "prev": 30.9, "acc": 69.8, "prec": 56.2, "rec": 9.9, "spec": 96.5, "f1": 16.9, "auc": 0.719},
            {"q": "Below 75 nmol/L (under the target)", "prev": 68.6, "acc": 74.9, "prec": 78.8, "rec": 86.6, "spec": 49.3, "f1": 82.6, "auc": 0.761},
        ],
        "flag": {"cut": 64, "prec": 44.1, "rec": 81.8},
        "acc3": 47.1, "base3": 37.7, "bal3": 45.0, "f13": 41.7,
        "groups": [
            {"name": "LOW (below 50)", "prec": 56.2, "rec": 9.9, "f1": 16.9},
            {"name": "MIDDLE (50 to 75)", "prec": 40.8, "rec": 75.8, "f1": 53.1},
            {"name": "HIGHER (75+)", "prec": 62.8, "rec": 49.3, "f1": 55.2},
        ],
        "cm": [[172, 1444, 114], [111, 1597, 399], [23, 869, 867]],
    },
    "sun": {
        "title": "Sun-habits model", "people": 3581, "inputs": "adds time outdoors and skin reaction (ages 20-59)",
        "mae": 17.4, "r2": 0.193, "w10": 36, "w20": 66,
        "cuts": [
            {"q": "Below 30 nmol/L (deficient)", "prev": 9.4, "acc": 90.6, "prec": 0.0, "rec": 0.0, "spec": 100.0, "f1": 0.0, "auc": 0.732},
            {"q": "Below 50 nmol/L (low)", "prev": 35.1, "acc": 68.8, "prec": 63.5, "rec": 26.3, "spec": 91.8, "f1": 37.1, "auc": 0.721},
            {"q": "Below 75 nmol/L (under the target)", "prev": 75.2, "acc": 76.8, "prec": 80.0, "rec": 92.3, "spec": 29.8, "f1": 85.7, "auc": 0.734},
        ],
        "flag": {"cut": 62, "prec": 47.6, "rec": 80.4},
        "acc3": 47.7, "base3": 40.1, "bal3": 44.5, "f13": 43.8,
        "groups": [
            {"name": "LOW (below 50)", "prec": 63.5, "rec": 26.3, "f1": 37.1},
            {"name": "MIDDLE (50 to 75)", "prec": 43.0, "rec": 77.5, "f1": 55.3},
            {"name": "HIGHER (75+)", "prec": 56.0, "rec": 29.8, "f1": 38.9},
        ],
        "cm": [[330, 889, 38], [153, 1113, 170], [37, 586, 265]],
    },
}}



MODELS = {}
try:
    import joblib
    import pandas as pd
    for key, fname in (("base", "vitd_model.joblib"), ("sun", "vitd_model_sun.joblib")):
        try:
            m = joblib.load(os.path.join(BASE, fname))
            if m.get("kind") != "regression":
                raise ValueError("not a regression model file: run the training script again")
            MODELS[key] = m
            print("ML model '%s' loaded: %s | MAE %.1f nmol/L | features %s"
                  % (key, m["name"], m["mae"], m["features"]))
        except Exception as e:
            print("ML model '%s' NOT loaded: %s" % (key, e))
except Exception as e:
    print("ML libraries missing:", e)


@lru_cache(maxsize=2048)
def _level(key, vals):
    m = MODELS[key]
    row = pd.DataFrame([dict(zip(m["features"], vals))])[m["features"]]
    return float(m["model"].predict(row)[0])


def rda_goal(age):
    """Standard recommended intake (IU/day): 600, and 800 above 70 years."""
    return 800 if age > 70 else 600


def predict_for(p):
    """Predicted serum 25(OH)D (nmol/L) and the daily IU goal. None if no model applies."""
    age = p["age"]
    if not MODELS or age < MODEL_MIN_AGE or age > MODEL_MAX_AGE:
        return None
    wd, we = p.get("outdoor_wd"), p.get("outdoor_we")
    has_sun = wd is not None and we is not None
    lo, hi = MODELS["sun"].get("age_range", [20, 59]) if "sun" in MODELS else (20, 59)
    key = "sun" if ("sun" in MODELS and has_sun and lo <= age <= hi) else "base"
    if key not in MODELS:
        return None
    m = MODELS[key]

    avg_out = (5 * wd + 2 * we) / 7.0 if has_sun else None
    outdoor = min(avg_out, 480.0) if has_sun else 0.0
    feat = {
        "age": float(age),
        "female": {"female": 1.0, "male": 0.0}.get(p["sex"], 0.5),
        "bmi": float(p["bmi"]),
        "supplement": 1.0 if p["supplement"] == "yes" else 0.0,
        "outdoor_min": outdoor,
        "skin_react": float(p["answers"][1] + 1),     # quiz question 2 -> NHANES DED031 scale 1..5
        "sunscreen": float({"always": 1, "sometimes": 3, "never": 5}.get(p["sunscreen"], 3)),
    }

    def lvl(f):
        return max(_level(key, tuple(f[c] for c in m["features"])), 0.0)

    level = lvl(feat)
    what_if = None
    if key == "sun":
        what_if = []
        for extra in (0, 30, 60, 120):
            f2 = dict(feat, outdoor_min=min(outdoor + extra, 480.0))
            what_if.append({"extra": extra, "level": round(lvl(f2), 1)})

    gap = max(TARGET_NMOL - level, 0.0)
    gap_iu = int(min(round(gap / NMOL_PER_100IU * 10) * 10, MAX_GOAL_IU))   # 1 nmol/L = 50 IU/day
    rda = rda_goal(age)
    goal = max(gap_iu, rda)          # never below the standard intake: the model error is large

    notes = []
    if key == "base" and has_sun and not (lo <= age <= hi):
        notes.append("Your 'time outdoors' answers are not used for your age: the survey behind the "
                     "sun model only covered ages %d to %d." % (lo, hi))
    if key == "base" and avg_out is not None and avg_out < 30:
        notes.append("You report very little time outdoors. The basic model cannot see this, "
                     "so it may overestimate your level.")

    if level < 50:
        cat, color = "Predicted level: LOW range (below 50)", "#fb923c"
    elif level < TARGET_NMOL:
        cat, color = "Predicted level: MIDDLE range (50-75)", "#fcd34d"
    else:
        cat, color = "Predicted level: HIGHER range (75+)", "#86efac"

    flag_cut = FLAG_CUT[key]
    flag_prec, flag_rec = FLAG_STATS[key]
    pct = lambda v: round(min(max(v, 0.0), 120.0) / 120.0 * 100, 1)     # position on the 0-120 scale
    g_lo, g_hi = pct(level - m["mae"]), pct(level + m["mae"])

    return {"level": round(level, 1), "low": round(max(level - m["mae"], 0), 1),
            "high": round(level + m["mae"], 1), "gap": round(gap, 1), "gap_iu": gap_iu,
            "rda": rda, "goal_iu": goal, "target": TARGET_NMOL, "mae": round(m["mae"], 1),
            "r2": round(m["r2"], 2), "rows": m.get("rows"), "cat": cat, "color": color,
            "model_used": key, "outdoor_min": round(outdoor),
            "skin_react": int(feat["skin_react"]), "what_if": what_if, "notes": notes,
            "flag": level < flag_cut, "flag_cut": flag_cut, "flag_prec": flag_prec, "flag_rec": flag_rec,
            "g_pos": pct(level), "g_lo": g_lo, "g_w": round(g_hi - g_lo, 1)}


def personal_goal(p):
    """Daily vitamin D goal (IU): the larger of the model's gap and the standard intake."""
    pr = predict_for(p)
    if pr is not None:
        return pr["goal_iu"]
    return rda_goal(p["age"])


def capacity_table(p, goal):
    """Research-based: what the user's own answers mean in minutes, for a few reference UV values."""
    med = MED[p["skin_type"]]
    spf = SPF_EFF[p["sunscreen"]]
    af = age_factor(p["age"])
    rows = []
    for uv in (4, 7, 10):
        rate = uv * UVI_TO_WM2 / spf                            # J/m2 per second after sunscreen
        iu_min = IU_FULL_MED * (rate * 60 / med) * p["exposed_frac"] * af
        safe_min = SAFE_FRAC * med / rate / 60                  # minutes until 80% of burn dose
        goal_min = goal / iu_min if iu_min > 0 else None
        rows.append({"uv": uv, "iu_min": round(iu_min, 1), "safe_min": round(safe_min),
                     "goal_min": round(goal_min) if goal_min is not None else None,
                     "ok": goal_min is not None and goal_min <= safe_min})
    return {"rows": rows, "goal": goal,
            "max_safe_iu": round(IU_FULL_MED * SAFE_FRAC * p["exposed_frac"] * af)}


# ---------------------------------------------------------------- sensor calibration
# UV_CAL = (UV index from a weather site) / (UV index shown on dashboard).
# Measure with a diffuser over the sensor so the mV is not clipped (< 2500 mV).
UV_CAL = 1.0

# ---------------------------------------------------------------- profile options
QUIZ = [
    ("Natural skin colour (a part that is never in the sun)",
     ["Very fair", "Fair", "Light brown / wheatish", "Brown", "Dark brown / black"]),
    ("After about 1 hour of midday sun with no protection, your skin...",
     ["Always burns badly and peels", "Usually burns", "Sometimes burns mildly",
      "Rarely burns", "Never burns"]),
    ("How does your skin tan?",
     ["Never tans, only burns", "Tans lightly", "Tans gradually to light brown",
      "Tans well to moderate brown", "Tans very easily to deep brown"]),
    ("Freckles on skin that is not exposed to sun",
     ["Many", "Several", "A few", "Very few", "None"]),
    ("How sensitive is your face to the sun?",
     ["Very sensitive", "Sensitive", "Normal", "Very resistant", "Never had a problem"]),
]

EXPOSED = {
    "face_hands": ("Face + hands (full sleeves, long pants)", 0.10),
    "arms": ("Face, hands + arms (half sleeves)", 0.25),
    "arms_legs": ("Arms + legs (shorts and t-shirt)", 0.40),
}


def skin_type_from_score(score):
    for t, limit in enumerate((3, 7, 11, 14, 17), start=1):
        if score <= limit:
            return t
    return 6


# ---------------------------------------------------------------- exposure model
UVI_TO_WM2 = 0.025          # UV index 1 = 25 mW/m2 erythemal irradiance
MED = {1: 200, 2: 250, 3: 350, 4: 450, 5: 600, 6: 900}   # J/m2 for mild redness (approx.)
SPF_EFF = {"always": 4.0, "sometimes": 1.5, "never": 1.0}  # effective, real-world use
SAFE_FRAC = 0.8             # warn at 80% of MED
IU_FULL_MED = 10000         # approx. IU made by 1 MED over the whole body
MAX_GAP_S = 15              # gap between readings counted at most this long
VITD_MIN_UVI = 3            # below this, vitamin D synthesis is minimal

# auto session rules
AUTO_START_UVI = 3.0
AUTO_START_N = 3            # this many readings in a row >= 3  -> start (~15 s)
AUTO_STOP_N = 6             # this many readings in a row < 3   -> stop  (~30 s)
OFFLINE_S = 60              # auto session ends if no reading for this long
MAX_SESSION_S = 3 * 3600    # forgotten sessions are capped at 3 h


def age_factor(age):
    return 1.0 if age <= 20 else max(0.5, 1 - 0.01 * (age - 20))


def metrics(q, raw_dose):
    """raw_dose (J/m2 at the sensor) -> personal numbers. q needs skin_type, sunscreen,
    exposed_frac, age."""
    eff = raw_dose / SPF_EFF[q["sunscreen"]]
    frac = eff / MED[q["skin_type"]]
    iu = IU_FULL_MED * min(frac, 1.0) * q["exposed_frac"] * age_factor(q["age"])
    return {"eff_dose": round(eff, 1), "pct_med": round(frac * 100, 1), "iu": round(iu)}


def integrate(c, start, end):
    rows = c.execute(
        "SELECT ts, uvi FROM readings WHERE ts >= ? AND ts <= ? ORDER BY ts", (start, end)
    ).fetchall()
    dose, mx, total = 0.0, 0.0, 0.0
    for i, r in enumerate(rows):
        nxt = rows[i + 1]["ts"] if i + 1 < len(rows) else end
        dt = min(max(nxt - r["ts"], 0.0), MAX_GAP_S)
        dose += r["uvi"] * UVI_TO_WM2 * dt
        mx = max(mx, r["uvi"])
        total += r["uvi"]
    avg = total / len(rows) if rows else 0.0
    return dose, avg, mx, len(rows)


def today_start():
    return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


# ---------------------------------------------------------------- database
def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with closing(db()) as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS users(
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                email         TEXT NOT NULL UNIQUE,
                name          TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at    REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS readings(
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        REAL NOT NULL,
                mv        REAL NOT NULL,
                uvi       REAL NOT NULL,
                saturated INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS profiles(
                user_id      INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                birth_year   INTEGER NOT NULL,
                sex          TEXT    NOT NULL,
                height_cm    REAL    NOT NULL,
                weight_kg    REAL    NOT NULL,
                quiz_answers TEXT    NOT NULL,
                quiz_score   INTEGER NOT NULL,
                skin_type    INTEGER NOT NULL,
                exposed      TEXT    NOT NULL,
                sunscreen    TEXT    NOT NULL,
                supplement   TEXT    NOT NULL,
                updated_at   REAL    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sun_sessions(
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                start_ts  REAL NOT NULL,
                end_ts    REAL,
                raw_dose  REAL NOT NULL DEFAULT 0,
                avg_uvi   REAL NOT NULL DEFAULT 0,
                max_uvi   REAL NOT NULL DEFAULT 0,
                pct_med   REAL NOT NULL DEFAULT 0,
                iu        REAL NOT NULL DEFAULT 0,
                skin_type INTEGER NOT NULL,
                sunscreen TEXT NOT NULL,
                exposed   TEXT NOT NULL
            );
            """
        )
        # small migrations for older databases
        for stmt in (
            "ALTER TABLE users ADD COLUMN auto_mode INTEGER NOT NULL DEFAULT 1",
            "ALTER TABLE users ADD COLUMN auto_hold INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE sun_sessions ADD COLUMN auto INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE profiles ADD COLUMN outdoor_wd INTEGER",     # minutes outdoors 9-5, work/school days
            "ALTER TABLE profiles ADD COLUMN outdoor_we INTEGER",     # minutes outdoors 9-5, other days
        ):
            try:
                c.execute(stmt)
            except sqlite3.OperationalError:
                pass  # column already exists
        c.commit()


def init_labs():
    with closing(db()) as c:
        c.execute(
            "CREATE TABLE IF NOT EXISTS lab_results("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,"
            "taken_on TEXT NOT NULL, value REAL NOT NULL, unit TEXT NOT NULL,"
            "value_nmol REAL NOT NULL, predicted_nmol REAL, model_used TEXT,"
            "features TEXT NOT NULL, consent INTEGER NOT NULL DEFAULT 0,"
            "created_at REAL NOT NULL)")
        c.commit()


def get_profile(user_id, c=None):
    sql = "SELECT * FROM profiles WHERE user_id = ?"
    if c is None:
        with closing(db()) as c2:
            row = c2.execute(sql, (user_id,)).fetchone()
    else:
        row = c.execute(sql, (user_id,)).fetchone()
    if row is None:
        return None
    p = dict(row)
    p["answers"] = [int(x) for x in p["quiz_answers"].split(",")]
    p["age"] = time.localtime().tm_year - p["birth_year"]
    p["bmi"] = round(p["weight_kg"] / (p["height_cm"] / 100) ** 2, 1)
    p["exposed_label"], p["exposed_frac"] = EXPOSED[p["exposed"]]
    return p


# ---------------------------------------------------------------- auth helpers
@app.before_request
def load_user():
    g.user = None
    uid = session.get("user_id")
    if uid is not None:
        with closing(db()) as c:
            g.user = c.execute(
                "SELECT id, email, name, auto_mode FROM users WHERE id = ?", (uid,)
            ).fetchone()
        if g.user is None:
            session.clear()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            if request.path.startswith("/api/"):
                return jsonify(error="login required"), 401
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


# ---------------------------------------------------------------- account pages
@app.get("/")
def index():
    return redirect(url_for("dashboard" if g.user else "login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        error = None
        if not name:
            error = "Please enter your name."
        elif not EMAIL_RE.match(email):
            error = "Please enter a valid email."
        elif len(password) < 8:
            error = "Password must be at least 8 characters."
        if error is None:
            try:
                with closing(db()) as c:
                    cur = c.execute(
                        "INSERT INTO users(email, name, password_hash, created_at) VALUES (?,?,?,?)",
                        (email, name, generate_password_hash(password), time.time()),
                    )
                    c.commit()
                    uid = cur.lastrowid
            except sqlite3.IntegrityError:
                error = "That email is already registered."
            else:
                session.clear()
                session["user_id"] = uid
                return redirect(url_for("profile"))
        flash(error)
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        with closing(db()) as c:
            row = c.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if row is None or not check_password_hash(row["password_hash"], password):
            flash("Wrong email or password.")
        else:
            session.clear()
            session["user_id"] = row["id"]
            return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------- profile + dashboard
def parse_profile_form(form):
    year_now = time.localtime().tm_year
    try:
        birth_year = int(form.get("birth_year", ""))
        height = float(form.get("height_cm", ""))
        weight = float(form.get("weight_kg", ""))
    except ValueError:
        return None, "Birth year, height and weight must be numbers."
    if not (year_now - 100 <= birth_year <= year_now - 10):
        return None, "Birth year looks wrong (age must be 10 to 100)."
    if not (100 <= height <= 230):
        return None, "Height must be between 100 and 230 cm."
    if not (20 <= weight <= 250):
        return None, "Weight must be between 20 and 250 kg."

    try:
        outdoor_wd = int(form.get("outdoor_wd", ""))
        outdoor_we = int(form.get("outdoor_we", ""))
    except ValueError:
        return None, "Minutes outdoors must be whole numbers."
    if not (0 <= outdoor_wd <= 480 and 0 <= outdoor_we <= 480):
        return None, "Minutes outdoors must be between 0 and 480 (9 am to 5 pm is 8 hours)."

    sex = form.get("sex", "")
    sunscreen = form.get("sunscreen", "")
    supplement = form.get("supplement", "")
    exposed = form.get("exposed", "")
    if sex not in ("female", "male", "other"):
        return None, "Please choose sex."
    if sunscreen not in ("always", "sometimes", "never"):
        return None, "Please choose how often you use sunscreen."
    if supplement not in ("yes", "no"):
        return None, "Please say whether you take vitamin D supplements."
    if exposed not in EXPOSED:
        return None, "Please choose how much skin is usually uncovered."

    answers = []
    for i in range(len(QUIZ)):
        v = form.get("q%d" % i, "")
        if v not in ("0", "1", "2", "3", "4"):
            return None, "Please answer all 5 skin questions."
        answers.append(int(v))
    score = sum(answers)
    return {
        "birth_year": birth_year, "sex": sex, "height_cm": height, "weight_kg": weight,
        "quiz_answers": ",".join(str(a) for a in answers), "quiz_score": score,
        "skin_type": skin_type_from_score(score), "exposed": exposed,
        "sunscreen": sunscreen, "supplement": supplement,
        "outdoor_wd": outdoor_wd, "outdoor_we": outdoor_we,
    }, None


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        values, error = parse_profile_form(request.form)
        if error:
            flash(error)
        else:
            values["user_id"] = g.user["id"]
            values["updated_at"] = time.time()
            cols = ", ".join(values)
            marks = ", ".join(":" + k for k in values)
            updates = ", ".join("%s = excluded.%s" % (k, k) for k in values if k != "user_id")
            with closing(db()) as c:
                c.execute(
                    "INSERT INTO profiles(%s) VALUES (%s) "
                    "ON CONFLICT(user_id) DO UPDATE SET %s" % (cols, marks, updates),
                    values,
                )
                c.commit()
            flash("Profile saved.", "ok")
            return redirect(url_for("dashboard"))
    return render_template("profile.html", p=get_profile(g.user["id"]),
                           quiz=QUIZ, exposed=EXPOSED, form=request.form)


@app.get("/dashboard")
@login_required
def dashboard():
    p = get_profile(g.user["id"])
    needs_update = bool(p and 20 <= p["age"] <= 59 and p.get("outdoor_wd") is None)
    cap = capacity_table(p, personal_goal(p)) if p else None
    return render_template("dashboard.html", p=p, pred=predict_for(p) if p else None,
                           needs_update=needs_update, cap=cap, report=MODEL_REPORT)


# ---------------------------------------------------------------- about + blood-test log
@app.get("/about")
def about():
    return render_template("about.html")


@app.route("/labs", methods=["GET", "POST"])
@login_required
def labs():
    p = get_profile(g.user["id"])
    if p is None:
        flash("Please set up your profile first.")
        return redirect(url_for("profile"))
    today = datetime.now().date()
    if request.method == "POST":
        error, d = None, None
        try:
            value = float(request.form.get("value", ""))
        except ValueError:
            value = None
        unit = request.form.get("unit", "")
        if value is None or not (1 <= value <= 500):
            error = "Enter the 25-OH vitamin D value from your lab report."
        elif unit not in ("nmol/L", "ng/mL"):
            error = "Choose the unit written on your report."
        else:
            try:
                d = datetime.strptime(request.form.get("taken_on", ""), "%Y-%m-%d").date()
            except ValueError:
                d = None
            if d is None or d > today or (today - d).days > 365:
                error = "Enter the test date (within the last year)."
        if error is None and request.form.get("consent") != "yes":
            error = "Please tick the consent box to save the result."
        if error:
            flash(error)
        else:
            nmol = value * 2.496 if unit == "ng/mL" else value        # 1 ng/mL = 2.496 nmol/L
            pr = predict_for(p)
            snap_ = {k: p[k] for k in ("age", "sex", "bmi", "supplement", "outdoor_wd", "outdoor_we",
                                       "skin_type", "exposed", "sunscreen", "quiz_answers")}
            snap_["skin_react"] = p["answers"][1] + 1
            with closing(db()) as c:
                c.execute(
                    "INSERT INTO lab_results(user_id, taken_on, value, unit, value_nmol, predicted_nmol,"
                    " model_used, features, consent, created_at) VALUES (?,?,?,?,?,?,?,?,1,?)",
                    (g.user["id"], d.isoformat(), value, unit, round(nmol, 1),
                     pr["level"] if pr else None, pr["model_used"] if pr else None,
                     json.dumps(snap_), time.time()))
                c.commit()
            flash("Saved. Thank you.", "ok")
            return redirect(url_for("labs"))
    with closing(db()) as c:
        rows = c.execute(
            "SELECT taken_on, value_nmol, predicted_nmol, model_used FROM lab_results "
            "WHERE user_id = ? ORDER BY taken_on DESC, id DESC", (g.user["id"],)).fetchall()
    return render_template("labs.html", rows=rows, today=today.isoformat())


# ---------------------------------------------------------------- session helpers
def get_active(c, uid):
    return c.execute(
        "SELECT * FROM sun_sessions WHERE user_id = ? AND end_ts IS NULL", (uid,)
    ).fetchone()


def snap(active, p):
    """Profile values as they were when the session started."""
    q = dict(p)
    q["skin_type"] = active["skin_type"]
    q["sunscreen"] = active["sunscreen"]
    q["exposed_frac"] = EXPOSED[active["exposed"]][1]
    return q


def start_session(c, uid, auto, start_ts=None):
    p = c.execute(
        "SELECT skin_type, sunscreen, exposed FROM profiles WHERE user_id = ?", (uid,)
    ).fetchone()
    if p is None or get_active(c, uid) is not None:
        return
    c.execute(
        "INSERT INTO sun_sessions(user_id, start_ts, skin_type, sunscreen, exposed, auto) "
        "VALUES (?,?,?,?,?,?)",
        (uid, start_ts or time.time(), p["skin_type"], p["sunscreen"], p["exposed"], auto),
    )


def finish_session(c, active, end_ts):
    p = get_profile(active["user_id"], c)
    end = max(min(end_ts, active["start_ts"] + MAX_SESSION_S), active["start_ts"])
    raw, avg, mx, n = integrate(c, active["start_ts"], end)
    m = metrics(snap(active, p), raw)
    c.execute(
        "UPDATE sun_sessions SET end_ts=?, raw_dose=?, avg_uvi=?, max_uvi=?, pct_med=?, iu=? "
        "WHERE id=?",
        (end, raw, avg, mx, m["pct_med"], m["iu"], active["id"]),
    )


def auto_update(c, uvi):
    """Called after every new reading: start / stop sessions for users with Auto ON."""
    users = c.execute(
        "SELECT u.id, u.auto_hold FROM users u JOIN profiles p ON p.user_id = u.id "
        "WHERE u.auto_mode = 1"
    ).fetchall()
    if not users:
        return
    last = [r["uvi"] for r in c.execute(
        "SELECT uvi FROM readings ORDER BY id DESC LIMIT ?", (AUTO_STOP_N,))]
    sun_now = len(last) >= AUTO_START_N and all(v >= AUTO_START_UVI for v in last[:AUTO_START_N])
    dark_now = len(last) >= AUTO_STOP_N and all(v < AUTO_START_UVI for v in last)
    for u in users:
        uid = u["id"]
        hold = u["auto_hold"]
        if uvi < AUTO_START_UVI and hold:
            c.execute("UPDATE users SET auto_hold = 0 WHERE id = ?", (uid,))
            hold = 0
        active = get_active(c, uid)
        if sun_now and active is None and not hold:
            first = c.execute(
                "SELECT ts FROM readings ORDER BY id DESC LIMIT 1 OFFSET ?",
                (AUTO_START_N - 1,)).fetchone()
            start_session(c, uid, 1, first["ts"] if first else None)
        elif dark_now and active is not None and active["auto"]:
            lastsun = c.execute(
                "SELECT ts FROM readings WHERE uvi >= ? ORDER BY id DESC LIMIT 1",
                (AUTO_START_UVI,)).fetchone()
            finish_session(c, active, lastsun["ts"] if lastsun else time.time())


# ---------------------------------------------------------------- session API
@app.post("/api/session/start")
@login_required
def session_start():
    if get_profile(g.user["id"]) is None:
        return jsonify(error="profile"), 400
    with closing(db()) as c:
        start_session(c, g.user["id"], 0)
        c.execute("UPDATE users SET auto_hold = 0 WHERE id = ?", (g.user["id"],))
        c.commit()
    return jsonify(ok=True)


@app.post("/api/session/stop")
@login_required
def session_stop():
    if get_profile(g.user["id"]) is None:
        return jsonify(error="profile"), 400
    with closing(db()) as c:
        active = get_active(c, g.user["id"])
        if active is not None:
            finish_session(c, active, time.time())
            last = c.execute("SELECT uvi FROM readings ORDER BY id DESC LIMIT 1").fetchone()
            if last and last["uvi"] >= AUTO_START_UVI:
                # user stopped on purpose: do not auto-restart until UV drops below 3
                c.execute("UPDATE users SET auto_hold = 1 WHERE id = ?", (g.user["id"],))
            c.commit()
    return jsonify(ok=True)


@app.post("/api/auto")
@login_required
def set_auto():
    on = 1 if (request.get_json(silent=True) or {}).get("on") else 0
    with closing(db()) as c:
        c.execute("UPDATE users SET auto_mode = ?, auto_hold = 0 WHERE id = ?",
                  (on, g.user["id"]))
        c.commit()
    return jsonify(ok=True, auto=bool(on))


def build_status(uid, auto_mode):
    """All personal numbers for one user (used by the dashboard and the OLED)."""
    p = get_profile(uid)
    if p is None:
        return None
    now = time.time()
    with closing(db()) as c:
        last = c.execute("SELECT uvi, ts FROM readings ORDER BY id DESC LIMIT 1").fetchone()
        active = get_active(c, uid)
        stale = bool(active and active["auto"] and (last is None or now - last["ts"] > OFFLINE_S))
        if active and (stale or now - active["start_ts"] > MAX_SESSION_S):
            finish_session(c, active, last["ts"] if (stale and last) else now)
            c.commit()
            active = None

        done = c.execute(
            "SELECT COALESCE(SUM(iu),0) AS iu, COALESCE(SUM(pct_med),0) AS pct, "
            "COALESCE(SUM(end_ts-start_ts),0) AS secs FROM sun_sessions "
            "WHERE user_id=? AND end_ts IS NOT NULL AND start_ts>=?",
            (uid, today_start()),
        ).fetchone()
        live = {"iu": 0, "pct_med": 0.0, "eff_dose": 0.0}
        elapsed = 0
        if active:
            raw, _, _, _ = integrate(c, active["start_ts"], now)
            live = metrics(snap(active, p), raw)
            elapsed = int(now - active["start_ts"])

    uvi = last["uvi"] if last else 0.0
    age_s = round(now - last["ts"], 1) if last else None
    today_iu = done["iu"] + live["iu"]
    today_pct = round(done["pct"] + live["pct_med"], 1)
    today_secs = int(done["secs"] + elapsed)

    med = MED[p["skin_type"]]
    spf = SPF_EFF[p["sunscreen"]]
    rate = uvi * UVI_TO_WM2 / spf                      # J/m2 per second after sunscreen
    remaining_pct = SAFE_FRAC * 100 - today_pct
    if remaining_pct <= 0:
        minutes_left = 0.0
    elif rate > 0.0005:
        minutes_left = round(remaining_pct / (rate / med * 100) / 60, 1)
    else:
        minutes_left = None
    iu_per_min = round(IU_FULL_MED * (rate * 60 / med) * p["exposed_frac"]
                       * age_factor(p["age"]), 1)

    goal = personal_goal(p)                            # <- comes from the trained model
    remaining_iu = max(goal - today_iu, 0)
    vitd_ok = uvi >= VITD_MIN_UVI and iu_per_min > 0
    need_min = round(remaining_iu / iu_per_min, 1) if vitd_ok else None
    total_min = round(goal / iu_per_min, 1) if vitd_ok else None

    return dict(
        uvi=round(uvi, 1), age_s=age_s, active=active is not None,
        auto_session=bool(active and active["auto"]), auto_mode=bool(auto_mode),
        elapsed_s=elapsed, session=live,
        today={"iu": round(today_iu), "pct_med": today_pct, "sun_s": today_secs},
        goal_iu=goal, remaining_iu=round(remaining_iu), need_min=need_min,
        total_min=total_min, minutes_left=minutes_left, iu_per_min=iu_per_min,
        vitd_ok=vitd_ok, warn=today_pct >= SAFE_FRAC * 100, skin_type=p["skin_type"],
    )


@app.get("/api/session/status")
@login_required
def session_status():
    s = build_status(g.user["id"], g.user["auto_mode"])
    if s is None:
        return jsonify(error="profile"), 400
    return jsonify(s)


@app.get("/api/device")
def device_state():
    """Small JSON for the ESP32 OLED: the user with a running session, else the latest user.
    (No login: the ESP32 cannot log in. Keep it on your own hotspot for now.)"""
    with closing(db()) as c:
        u = c.execute(
            "SELECT u.id, u.name, u.auto_mode FROM users u "
            "JOIN profiles p ON p.user_id = u.id "
            "LEFT JOIN sun_sessions s ON s.user_id = u.id "
            "GROUP BY u.id ORDER BY MAX(CASE WHEN s.id IS NULL THEN 0 "
            "WHEN s.end_ts IS NULL THEN 1e18 ELSE s.end_ts END) DESC LIMIT 1"
        ).fetchone()
    if u is None:
        return jsonify(empty=True)
    s = build_status(u["id"], u["auto_mode"])
    return jsonify(
        name=u["name"], uvi=s["uvi"], active=s["active"], auto_session=s["auto_session"],
        elapsed_s=s["elapsed_s"], today_iu=s["today"]["iu"], goal_iu=s["goal_iu"],
        remaining_iu=s["remaining_iu"], need_min=s["need_min"],
        minutes_left=s["minutes_left"], vitd_ok=s["vitd_ok"], warn=s["warn"],
    )


@app.get("/api/sessions")
@login_required
def sessions_list():
    with closing(db()) as c:
        rows = c.execute(
            "SELECT start_ts, end_ts, avg_uvi, max_uvi, pct_med, iu, auto FROM sun_sessions "
            "WHERE user_id=? AND end_ts IS NOT NULL ORDER BY id DESC LIMIT 60",
            (g.user["id"],),
        ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.get("/api/daily")
@login_required
def daily():
    p = get_profile(g.user["id"])
    if p is None:
        return jsonify(error="profile"), 400
    try:
        n = int(request.args.get("days", 7))
    except ValueError:
        n = 7
    if n not in (7, 14, 30):
        n = 7
    today = datetime.now().date()
    days = [today - timedelta(days=i) for i in range(n - 1, -1, -1)]
    first = datetime.combine(days[0], datetime.min.time()).timestamp()
    agg = {d: {"iu": 0.0, "min": 0.0, "n": 0, "uv_w": 0.0, "uv_max": 0.0} for d in days}
    with closing(db()) as c:
        rows = c.execute(
            "SELECT start_ts, end_ts, iu, avg_uvi, max_uvi FROM sun_sessions "
            "WHERE user_id=? AND end_ts IS NOT NULL AND start_ts>=?",
            (g.user["id"], first),
        ).fetchall()
    for r in rows:
        d = datetime.fromtimestamp(r["start_ts"]).date()
        if d in agg:
            mins = (r["end_ts"] - r["start_ts"]) / 60
            a = agg[d]
            a["iu"] += r["iu"]
            a["min"] += mins
            a["n"] += 1
            a["uv_w"] += r["avg_uvi"] * mins
            a["uv_max"] = max(a["uv_max"], r["max_uvi"])
    out = []
    for d in days:
        a = agg[d]
        out.append({"date": d.isoformat(), "label": d.strftime("%a %d"), "iu": round(a["iu"]),
                    "min": round(a["min"], 1), "sessions": a["n"],
                    "avg_uvi": round(a["uv_w"] / a["min"], 1) if a["min"] > 0 else 0,
                    "max_uvi": round(a["uv_max"], 1), "today": d == today})
    return jsonify(goal=personal_goal(p), days=out)


# ---------------------------------------------------------------- sensor API
@app.post("/api/uv")
def add_reading():
    """Called by the ESP32."""
    data = request.get_json(force=True, silent=True) or {}
    try:
        mv = float(data["mv"])
        uvi = float(data["uvi"])
        sat = 1 if data.get("saturated") else 0
    except (KeyError, TypeError, ValueError):
        return jsonify(error="JSON must contain numeric 'mv' and 'uvi'"), 400
    uvi = round(uvi * UV_CAL, 2)
    with closing(db()) as c:
        c.execute(
            "INSERT INTO readings(ts, mv, uvi, saturated) VALUES (?,?,?,?)",
            (time.time(), mv, uvi, sat),
        )
        c.commit()
        auto_update(c, uvi)
        c.commit()
    return jsonify(ok=True)


@app.get("/api/latest")
@login_required
def latest():
    with closing(db()) as c:
        row = c.execute("SELECT * FROM readings ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        return jsonify(empty=True)
    d = dict(row)
    d["age_s"] = round(time.time() - d["ts"], 1)
    return jsonify(d)


@app.get("/api/history")
@login_required
def history():
    n = min(int(request.args.get("n", 50)), 500)
    with closing(db()) as c:
        rows = c.execute(
            "SELECT ts, mv, uvi, saturated FROM readings ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
    return jsonify([dict(r) for r in reversed(rows)])


if __name__ == "__main__":
    init_db()
    init_labs()
    app.run(host="0.0.0.0", port=5000, debug=False)