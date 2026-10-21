# Personalised UV Index and Vitamin D Tracker

**Project description, research summary, and patient-facing explanation**

**An ESP32 + UV sensor + Flask web app + machine-learning project**

Dataset: US NHANES 2013-2014 | Hardware: ESP32, GUVA-S12SD, SSD1306 OLED | Software: MicroPython, Python, Flask, SQLite, scikit-learn

---

## 1. Overview

This project measures the live UV index with a low-cost sensor and turns it into **personal vitamin D advice**. Many people do not know how long they should stay in the sun to make enough vitamin D, or when the sun becomes a burn risk. The answer depends on the person (skin type, age, clothing, supplements, body weight) and on the UV level right now.

The system has three parts:

1. **Device (ESP32 + sensor + OLED):** reads UV every few seconds, shows it on a small screen, and sends it over WiFi to the server.
2. **Server (Flask + SQLite):** stores users, profiles and readings, runs the sun-exposure calculations, runs the trained ML model, and serves the dashboard.
3. **Machine-learning model (trained on NHANES):** predicts a person's blood vitamin D level (25-hydroxy vitamin D, in nmol/L) from their profile. The gap to a healthy target decides the daily vitamin D goal.

The dashboard provides an educational estimate and exposure-planning information: current UV,
an approximate session contribution, a model-based serum vitamin D estimate, and a conservative
burn-limit warning. It is not a blood test, diagnosis, prescription, or substitute for advice
from a qualified clinician.

### Short abstract

This project presents a low-cost Internet-of-Things prototype for combining live ultraviolet
(UV) sensing with a web-based vitamin D education and tracking dashboard. An ESP32 equipped
with a GUVA-S12SD analog UV sensor measures UV index readings and sends them to a Flask
application. The application stores readings and user-confirmed sessions in SQLite, estimates
UV dose and exposure risk, and uses Gradient Boosting regression models trained on NHANES
2013-2014 data to estimate serum 25-hydroxyvitamin D (`LBXVIDMS`) in nmol/L. A basic model
uses age, sex, body mass index (BMI), and supplement use; an optional sun-habits model adds
outdoor exposure and skin reaction.

In saved-artifact verification, the basic model achieved MAE 18.04 nmol/L, RMSE 23.81 nmol/L,
and R² 0.255 on 5,596 rows. The sun-habits model achieved MAE 17.39 nmol/L, RMSE
22.74 nmol/L, and R² 0.192 on 3,541 rows. A separate nested out-of-fold screening evaluation
used the medical definition `LBXVIDMS < 30 nmol/L` and selected prediction cutoffs of 54.5
nmol/L (basic) and 51.1 nmol/L (sun-habits). These are screening operating points, not
diagnostic thresholds. The system has not been validated on Indian patients and should not be
used to make treatment decisions.

### Intended audiences

- **Project paper or dissertation:** use the abstract, methods, results, limitations, and
  reproducibility sections as a structured description of the work.
- **Project presentation:** explain the sensor-to-dashboard data flow, the model inputs, and
  the measured validation results.
- **Patient or participant:** read the patient-facing section before using any estimate. It
  explains what the numbers mean, what they do not mean, and when a blood test or clinician is
  needed.


---

## 2. Objectives

- Build a working UV index meter from an ESP32 and an analog UV sensor.
- Support **multiple users**, each with a login and a personal profile stored in a database.
- Turn UV readings into a **personal** vitamin D estimate and a personal burn limit (no fake or default profile data).
- Train a model on a real public dataset (NHANES) and report its accuracy honestly.
- Show everything on a web dashboard and on the small OLED screen.

---

## 3. System architecture

```
 +--------------------+      WiFi (HTTP/JSON)      +--------------------------------+
 |  ESP32 (MicroPython)| --- POST /api/uv  ------> |  Flask server (laptop)         |
 |  GUVA-S12SD sensor  |                           |   - login, profiles, sessions  |
 |  SSD1306 OLED       | <--- GET /api/device ---- |   - sun-exposure model         |
 +--------------------+                            |   - ML model (joblib files)    |
                                                   |   - SQLite database            |
                                                   +---------------+----------------+
                                                                   |
                                                         browser (dashboard)
```

Data flow:

1. The ESP32 measures UV, calculates the UV index, and POSTs `{mv, uvi, saturated}` to `/api/uv` every 5 seconds.
2. The server saves each reading in the `readings` table and checks whether any user's **auto sun session** should start or stop.
3. The dashboard (in the browser) asks the server for the latest UV, the user's session numbers, charts, and the ML prediction.
4. The ESP32 asks `/api/device` for the personal vitamin D numbers and shows them on the OLED (page 2).

Both the ESP32 and the laptop must be on the same WiFi network (a phone hotspot was used). USB to the ESP32 is only for power and for uploading code; sensor data goes over WiFi.

---

## 4. Hardware

### 4.1 Components

| Component | Purpose | Notes |
|---|---|---|
| ESP32 development board | Microcontroller with built-in WiFi | Runs MicroPython; powered by USB |
| GUVA-S12SD UV sensor module | Measures UV (analog voltage) | Responds to roughly 240-370 nm UV light |
| 128x64 SSD1306 OLED display (I2C) | Shows UV index and vitamin D progress | I2C address `0x3C` |
| Jumper wires, breadboard | Connections | |
| Phone WiFi hotspot (2.4 GHz) + laptop | Network and server | ESP32 supports 2.4 GHz only |

### 4.2 Wiring

| From | To (ESP32) |
|---|---|
| Sensor VCC | 3V3 |
| Sensor GND | GND |
| Sensor SIG (analog out) | GPIO 34 (D34, ADC1) |
| OLED VCC | 3V3 |
| OLED GND | GND |
| OLED SDA | GPIO 21 (D21) |
| OLED SCL | GPIO 22 (D22) |

GPIO 34 is an input-only ADC1 pin, which keeps working while WiFi is on.

### 4.3 How the sensor reading becomes a UV index

The sensor gives a voltage that grows with UV. On the ESP32:

- The ADC is set to 11 dB attenuation (about 0-3.1 V range).
- **Noise reduction:** each reading is the average of 32 samples; five such readings are taken and the **median** is used.
- **Conversion:** `UVI = (mV - BASE) / FACTOR`, limited to the range 0 to 7.
  - `BASE = 142` mV (reading in the dark or indoors)
  - `FACTOR = 375` mV per UV index (calibrated value)
- **Saturation:** at about 3000 mV or more the ADC is clipped. The sensor then cannot show the true UV, so the reading is marked `saturated` and shown as **7+**.
- **Server calibration:** the server multiplies the UV index by `UV_CAL` (default 1.0). The idea is `UV_CAL = UV index from a weather site / UV index shown on the dashboard`, measured with a white-paper diffuser over the sensor so the voltage does not clip.

UV categories shown to the user: Low (below 3), Moderate (3 to 6), High (6 to 8), Very high (8 and above). Below UV 3 almost no vitamin D is made, so the app treats UV below 3 as "no vitamin D now".

### 4.4 ESP32 firmware (`main.py`, MicroPython)

- Connects to WiFi in station mode.
- Every 5 seconds: measure UV, show it, POST it to the server.
- Reads the personal numbers from `/api/device` and alternates two OLED pages every 4 seconds:
  - **Page 1:** UV index (large), category (Low/Moderate/High), and DARK / SHADE / SUNNY label.
  - **Page 2:** user name, session mode (AUTO / MANUAL / IDLE), a progress bar of today's vitamin D against the goal, minutes still needed, safe time left, and a warning "GO TO SHADE" when the burn limit is near.
- Libraries on the ESP32: `machine` (Pin, ADC, I2C), `network`, `framebuf`, `time`, `urequests`/`requests`, and `ssd1306` (driver file saved on the board).

---

## 5. Software stack and libraries

| Layer | Tool / library | Used for |
|---|---|---|
| ESP32 | MicroPython, Thonny | Firmware and uploading code |
| Backend | Python, **Flask** | Web server, pages and JSON API |
| Database | **SQLite** (`sqlite3`, built into Python) | Users, profiles, readings, sessions, lab results |
| Security | **Werkzeug** (`generate_password_hash`, `check_password_hash`) | Password hashing; secret key in `secret.key` |
| Templates | **Jinja2** (comes with Flask) | HTML pages |
| Frontend | HTML, CSS, plain JavaScript, inline **SVG** charts | Dashboard (no internet or chart library needed) |
| ML | **pandas**, **numpy**, **scikit-learn**, **joblib** | Data preparation, training, saving and loading models |
| Report charts | **matplotlib** | Predicted-vs-actual, ROC and other figures |
| Editor | VS Code, Thonny | Development |

### Project folder (server side)

```
vitd_project/
  server.py                  Flask app (all routes, models, exposure maths)
  tracker.db                 SQLite database (created automatically)
  secret.key                 Session secret (created automatically)
  vitd_model.joblib          Basic model (age 18-80, 4 inputs)
  vitd_model_sun.joblib      Sun-habits model (age 20-59, 6 inputs)
  data/                      NHANES .XPT files
  download_data.py           Downloads NHANES files
  train_model.py             Trains the basic model
  (sun model training script)
  check_model.py             Sanity checks of the models
  check_metrics.py           Accuracy, precision, recall, AUC
  make_charts.py             Report charts -> charts/
  templates/                 base, login, register, profile, dashboard, labs, about
```

---

## 6. Database design (SQLite)

| Table | Main columns | Purpose |
|---|---|---|
| `users` | id, email (unique), name, password_hash, created_at, auto_mode, auto_hold | Accounts and auto-session switch |
| `profiles` | user_id, birth_year, sex, height_cm, weight_kg, quiz_answers, quiz_score, skin_type, exposed, sunscreen, supplement, outdoor_wd, outdoor_we, updated_at | One personal profile per user |
| `readings` | id, ts, mv, uvi, saturated | Every UV reading from the ESP32 |
| `sun_sessions` | id, user_id, start_ts, end_ts, raw_dose, avg_uvi, max_uvi, pct_med, iu, skin_type, sunscreen, exposed, auto | One row per sun session |
| `lab_results` | id, user_id, taken_on, value, unit, value_nmol, predicted_nmol, model_used, features, consent, created_at | Optional blood-test results (saved only with consent) |

Small migrations add new columns to older databases without losing data.

---

## 7. User inputs (profile form)

| Input | Type / allowed values | Used for |
|---|---|---|
| Name, email, password (min 8 characters) | Text | Account |
| Birth year | Age must be 10 to 100 | Age (ML model, vitamin D production) |
| Sex | female / male / other | ML model |
| Height (100-230 cm), weight (20-250 kg) | Number | BMI (ML model) |
| Vitamin D supplement | yes / no | ML model |
| Minutes outdoors 9 am-5 pm on **work/school days** and on **other days** | 0-480 minutes each | ML sun model (average per day = (5 x work + 2 x other) / 7) |
| Sunscreen use | always / sometimes / never | Burn limit and vitamin D calculation |
| Usually uncovered skin | face + hands (10%), face + hands + arms (25%), arms + legs (40%) | Vitamin D calculation |
| 5 skin questions (score 0-4 each, total 0-20) | Multiple choice | Skin type 1 to 6 |

**The 5 skin questions:** natural skin colour; what happens after about 1 hour of unprotected midday sun; how the skin tans; freckles on skin not exposed to sun; how sensitive the face is to the sun.

**Skin type from total score:** 0-3 = type 1, 4-7 = type 2, 8-11 = type 3, 12-14 = type 4, 15-17 = type 5, 18-20 = type 6 (modelled on the Fitzpatrick scale).

Answer to question 2 (skin reaction to 1 hour of sun) is also converted to the NHANES scale (1 = burns badly ... 5 = never burns) and used as an ML feature.

---

## 8. Dataset: NHANES 2013-2014

NHANES (National Health and Nutrition Examination Survey, US CDC) combines interviews, body measurements and blood tests. The 2013-2014 cycle ("H" files) includes measured vitamin D. Files are downloaded from the CDC website and read with `pandas.read_sas` (`.XPT` format).

| File | Columns used | Meaning |
|---|---|---|
| `DEMO_H` | `SEQN`, `RIDAGEYR`, `RIAGENDR` | Person ID, age in years, sex (1 = male, 2 = female). About 10,175 people. |
| `BMX_H` | `BMXBMI` | Body mass index |
| `VID_H` | `LBXVIDMS` | **Target:** serum 25-hydroxy vitamin D (D2 + D3) in **nmol/L** |
| `DSQTOT_H` | `DSQTVD` | Total vitamin D from dietary supplements (mcg). Missing means none |
| `DEQ_H` | `DED120`, `DED125`, `DEQ034D`, `DED031` | Dermatology / sun questionnaire (3,928 people, ages 20-59) |

**`DEQ_H` columns:**

- `DED120`: minutes outdoors 9 am-5 pm (without shade) on work/school days.
- `DED125`: same, on non-work days.
- `DEQ034D`: sunscreen use (1 = always ... 5 = never; code 6 "does not go out in the sun" treated as 5).
- `DED031`: skin reaction after half an hour of sun following months of no sun (1 = burns badly ... 5 = nothing happens).

**Cleaning rules**

- Adults only (age 18 or more); drop rows with missing BMI or vitamin D; supplement missing = 0.
- `DED120` / `DED125`: values 3333, 7777, 9999 are **codes, not minutes**; only values up to 480 are kept. If one of the two is missing (for example the person has no work days), the other is used.
- `DEQ034D` and `DED031`: only valid codes 1-5 kept.

**Engineered features**

| Feature | Built from |
|---|---|
| `age` | `RIDAGEYR` |
| `female` | 1 if `RIAGENDR` = 2, else 0 |
| `bmi` | `BMXBMI` |
| `supplement` | 1 if `DSQTVD` > 0, else 0 |
| `outdoor_min` | (5 x `DED120` + 2 x `DED125`) / 7 |
| `skin_react` | `DED031` (1-5) |
| `sunscreen` | `DEQ034D` (1-5), tested but not used in the final model |

**Rows:** 5,596 adults for the basic model (ages 18-80); 3,541 people aged 20-59 for the sun model.

---

## 9. Machine learning

### 9.1 How the approach developed

1. **First attempt: classification** (is the person "low in vitamin D?") with logistic regression and random forest. AUC was about 0.73. Checking the CDC documentation then showed that `LBXVIDMS` is in **nmol/L**, not ng/mL, so the first threshold was wrong and was corrected.
2. **Switch to regression:** predicting the level itself in nmol/L is more useful, because the gap to a target gives an actual daily vitamin D goal.
3. **Basic model:** 4 features. Linear regression, random forest and gradient boosting were compared. All three gave almost the same error (MAE 18.0 to 18.2), so the limit is the data (few features), not the algorithm. Gradient boosting was saved.
4. **Sun-habits model:** the 5th NHANES file (`DEQ_H`) adds time outdoors, skin reaction and sunscreen. An ablation study (below) chose which to keep.

### 9.2 Final model settings

`GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8)`, evaluated with 5-fold cross-validation repeated with different random seeds, and saved with `joblib` together with its feature list, MAE, RMSE and R2.

### 9.3 Ablation: which sun feature helps?

(sun-model rows; MAE in nmol/L, lower is better)

| Features | MAE | R2 |
|---|---|---|
| Base (4 features) | 17.96 | 0.152 |
| + minutes outdoors | 17.81 | 0.165 |
| + skin reaction | 17.55 | 0.180 |
| + sunscreen | 17.69 | 0.169 |
| **+ minutes outdoors + skin reaction (chosen)** | **17.39** | **0.192** |
| + all three | 17.21 | 0.203 |

Sunscreen was left out because its gain was tiny and its effect in survey data is confounded (people who use sunscreen differ in other ways). It is still used in the physics part of the app.

### 9.4 The two models in the app

| | Basic model | Sun-habits model |
|---|---|---|
| Ages | 18-80 | 20-59 |
| Inputs | age, sex (female), BMI, supplement | + average minutes outdoors, skin reaction (1-5) |
| People | 5,596 | 3,541 |
| MAE | about 18.0 nmol/L | about 17.4 nmol/L |
| R2 | 0.255 | 0.192 |

The app uses the sun model when the user is 20-59 **and** has answered the outdoor-time questions; otherwise it uses the basic model. For ages below 18 or above 80 there is no prediction, and the standard recommended intake is used.

R2 values of the two models are not comparable because they use different groups of people. The ablation table above compares fairly on the same 3,541 people.

### 9.5 Validation (cross-validated, honest numbers)

| Measure | Basic | Sun |
|---|---|---|
| MAE (nmol/L) | 18.6 | 17.4 |
| R2 | 0.243 | 0.193 |
| Within +/-10 nmol/L | 34% | 36% |
| Within +/-20 nmol/L | 63% | 66% |
| ROC AUC, "below 50 nmol/L" | 0.719 | 0.721 |
| ROC AUC, "below 75 nmol/L" | 0.761 | 0.734 |
| 3-group accuracy (below 50 / 50-75 / 75+) | 47.1% (baseline 37.7%) | 47.7% (baseline 40.1%) |

- The calibration check shows that people grouped by predicted level have actual levels that rise in the same order.
- All 22 automatic sanity checks passed (directions of each input, extreme inputs, full server pipeline).
- **Direction of effects (all agree with research):** supplement raises the level (about +14 to +17 nmol/L), higher BMI lowers it, more time outdoors raises it, and people whose skin burns less show lower measured levels in US data.
- **Weakness:** predictions are pulled towards the population average. The original fixed-30
  regression cutoff produced no predicted deficient cases in the baseline evaluation, so
  precision, recall, and F1 were 0% at that operating point. A separate nested evaluation
  selected higher screening cutoffs to improve recall: 54.5 nmol/L for the basic model and
  51.1 nmol/L for the sun-habits model. These cutoffs produce more false positives and are
  not diagnostic thresholds. The models are useful for risk ranking and education, not for
  diagnosing deficiency.

### 9.6 From prediction to the daily goal

1. Predicted level (nmol/L) from the model.
2. **Gap** = 75 - predicted level (target 75 nmol/L, never below 0).
3. **IU to close the gap:** about 2 nmol/L rise per 100 IU a day (from published trials), so 1 nmol/L is about 50 IU/day; capped at 2000 IU.
4. **Daily goal** = the larger of the model's gap in IU and the standard recommended intake (600 IU, 800 IU above age 70). The goal is never set below the standard intake because the model error is large.

### 9.7 How the result is shown (honest wording)

- Label: "Predicted level: LOW range (below 50) / MIDDLE range (50-75) / HIGHER range (75+)", plus a colour scale with the likely range (+/- typical error).
- **"Higher risk of low vitamin D" screening flag** when the predicted level is below the
  selected nested-validation cutoff: 54.5 nmol/L for the basic model or 51.1 nmol/L for
  the sun-habits model. These cutoffs are intended to improve screening recall for the
  medical outcome below 30 nmol/L, at the cost of false positives. Not being flagged does
  not mean the level is adequate.
- A reminder that predictions are pulled towards the average and that only a blood test can confirm.

---

### 9.8 Deficiency screening results

The medical outcome is always defined from the measured laboratory value:

```text
Actual deficient = LBXVIDMS < 30 nmol/L
```

The prediction cutoff is a separate operating choice. It is used only to raise a
screening-style alert when the estimated level is below the selected cutoff. The latest
leakage-safe nested validation report (`deficiency_validation_report.json`) gives:

| Model and screening cutoff | Actual deficient | Predicted deficient | TP | FP | FN | TN | Precision | Recall | Specificity | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Basic regression, <54.5 nmol/L | 452 | 1,204 | 202 | 1,002 | 250 | 4,142 | 16.8% | 44.7% | 80.5% | 24.4% | 0.173 | 0.718 |
| Sun-habits regression, <51.1 nmol/L | 338 | 678 | 141 | 537 | 197 | 2,706 | 20.8% | 41.7% | 83.4% | 27.8% | 0.216 | 0.727 |

The selected cutoffs were chosen inside the training data using inner three-fold
out-of-fold predictions, with F1 as the primary objective and recall as the tie-break.
The five untouched outer folds were used for the reported results. Classifier comparisons
were also evaluated without allowing outer-fold labels to influence training or threshold
selection. In this run, no tested dedicated classifier demonstrated a defensible
simultaneous improvement in both precision and recall, so the saved regression artifacts
remain the production baseline.

The fixed 30 nmol/L prediction cutoff remains useful as a clearly labelled baseline, but
its 0% precision and recall must not be confused with the tuned screening results above.
Accuracy alone is not sufficient here because the deficient class is uncommon: predicting
"not deficient" for everyone can produce high accuracy while missing every deficient case.

### 9.9 Health and milk adjustment layer

The trained Gradient Boosting models use only their documented core features. Health answers
are handled once, after the original model prediction, in a separate configurable layer to
reduce double-counting risk. The dashboard keeps the original prediction, each contribution,
the total adjustment, and the final bounded prediction separate.

The current binary-health deductions are positive magnitudes in
`health_adjustments.json` and are subtracted only when the user selects **Yes**:

| Input | Experimental deduction (nmol/L) |
|---|---:|
| Kidney disease | 1.0 |
| Liver disease | 0.5 |
| Thyroid problems | 1.5 |
| Diabetes | 1.0 |
| Weak bones | 0.5 |
| Smoking | 1.0 |

The combined binary-health deduction is capped at **4.0 nmol/L**, and the kidney/diabetes
overlap rule prevents the configured overlap from being counted twice. No deduction is
applied for **No**, missing, or unrecognised answers. These are project-specific academic
heuristics, not NHANES-validated clinical effect sizes and not treatment recommendations.
The raw residual analyses are retained separately from the active demonstration settings.

Milk intake is handled independently. The current academic demonstration tiers are:

| Milk response | Experimental bonus (nmol/L) |
|---|---:|
| Never | 0.0 |
| Rarely (less than once a week) | 1.0 |
| Sometimes (once a week or more) | 1.5 |
| Every day | 2.5 |

The bonus is applied only when the recognised condition is satisfied. Unknown answers receive
zero. It must not be interpreted as proof that all milk raises vitamin D: fortification,
serving size, total diet, and individual biology vary. Both the health deductions and milk
bonus are labelled as experimental in the interface.

---

## 10. Sun-exposure (physics-style) model

This part does not come from the dataset. It uses published rules of thumb and the user's answers.

| Constant | Value | Meaning |
|---|---|---|
| UV index 1 | 0.025 W/m2 | Erythemal irradiance per UV index unit |
| MED by skin type 1-6 | 200, 250, 350, 450, 600, 900 J/m2 | Dose for mild redness |
| Effective sunscreen factor | always 4.0, sometimes 1.5, never 1.0 | Real-world protection |
| Burn warning | 80% of MED | Safe fraction per day |
| Vitamin D at 1 MED, whole body | about 10,000 IU | Used with exposed-skin fraction |
| Age factor | 1.0 up to 20 years, then minus 1% per year, minimum 0.5 | Skin makes less vitamin D with age |
| Minimum UV for vitamin D | 3 | Below this, synthesis is minimal |

**Calculation**

- **Dose:** each reading adds `UVI x 0.025 x seconds` (a gap between readings counts at most 15 s). Dose after sunscreen = dose / sunscreen factor.
- **% of burn limit** = dose / MED of the user's skin type.
- **Vitamin D made** = 10,000 x min(fraction of MED, 1) x exposed-skin fraction x age factor.
- **Minutes still needed** = remaining IU / IU per minute at the current UV; **safe time left** = minutes until 80% of MED.
- **Personal sun table:** for UV 4, 7 and 10, shows IU per minute, minutes for the goal, the safe limit, and the maximum vitamin D safely possible in a day with the user's usual clothing.

**Sun sessions**

- **Manual:** start / stop buttons.
- **Auto mode:** starts when UV stays at 3 or above for 3 readings in a row (about 15 s); stops when UV stays below 3 for 6 readings in a row (about 30 s). An auto session also ends if no reading arrives for 60 s, and no session runs longer than 3 hours. If the user stops manually while the sun is still strong, auto does not restart until UV drops below 3.
- A session keeps the skin type, sunscreen and clothing values that were in the profile when it started.

---

## 11. Web application

### Pages

| Page | Content |
|---|---|
| Register / Login | Account creation and login (hashed passwords, sessions) |
| Profile | The inputs in section 7, with validation |
| **Dashboard** | Live UV and chart, today's vitamin D ring, session control, burn-limit bar, ML prediction card with scale and flag, what-if table (extra time outdoors), personal sun table, 7-day chart, past sessions, profile summary |
| Blood-test log (`/labs`) | User can save a real lab value (nmol/L or ng/mL) with a consent tick; stored next to the model's prediction |
| About (`/about`) | How the numbers are made |

### API

| Endpoint | Method | Login | Purpose |
|---|---|---|---|
| `/api/uv` | POST | No (ESP32) | Receive a reading (`mv`, `uvi`, `saturated`) |
| `/api/device` | GET | No (ESP32) | Small JSON for the OLED |
| `/api/latest`, `/api/history` | GET | Yes | Latest reading, recent readings |
| `/api/session/start`, `/stop`, `/status` | POST/GET | Yes | Sun session control and personal numbers |
| `/api/auto` | POST | Yes | Switch auto mode |
| `/api/sessions`, `/api/daily` | GET | Yes | Past sessions, 7-day totals |

Dashboard refresh: live UV and session numbers every 2 s, UV chart every 5 s, history every 15 s, 7-day chart every 30 s.

---

## 12. Challenges and how they were handled

- **Sensor saturation:** a high UV day clipped the ADC at about 3100 mV, so the dashboard showed 7+ while a weather site said about 4. Fixed by marking saturated readings, and by the `UV_CAL` calibration with a diffuser.
- **WiFi / IP problems:** hotspot changes the laptop IP, so the ESP32 `SERVER_URL` must be updated.
- **Wrong unit assumption:** the vitamin D target was first treated as ng/mL; checking the documentation showed nmol/L, and the thresholds were corrected.
- **Survey codes mistaken for minutes:** values 3333 / 7777 / 9999 in the outdoor-time columns inflated the average to 558 minutes; they were removed. Handling "no workdays" correctly recovered 505 more people.
- **Weak predictive power:** instead of hiding it, the app reports typical error and uses wording and a flag that match what the model can really do.

---

## 13. Limitations

- Models were trained on **US adults (2013-2014)**, not on Indian users; effects such as skin reaction are an extrapolation.
- Only 4-6 inputs: diet, genetics, latitude and season are not in the model. R2 is 0.19-0.25 and the typical error is about 17-18 nmol/L.
- Outdoor time and skin reaction are self-reported.
- Decision-tree predictions are not smooth lines; small steps in the what-if table should not be over-read. Skin reaction codes 1, 2 and 3 give almost the same prediction.
- In US data, older age and "being in the supplement group" go with higher levels; an older user without supplements may be shown a better level than is true.
- The sun model covers ages 20-59 only; there is no prediction below 18 or above 80.
- The UV sensor is a hobby-grade analog sensor with a limited range and is not certified; exposure maths uses approximate constants.
- `/api/uv` and `/api/device` are open (no device key), so use only on a trusted network.
- Not a medical device: no diagnosis, no treatment advice.

### 13.1 Patient-facing explanation

**What this dashboard can do**

- Show the UV index measured by the connected sensor.
- Help record sun-exposure sessions and display an approximate exposure and burn-risk
  calculation.
- Provide a statistical estimate of serum 25(OH)D based on selected profile information.
- Show an educational screening alert when the estimate is below the selected model cutoff.

**What the number does not mean**

- It is not your measured vitamin D blood level.
- It cannot confirm or rule out deficiency.
- It does not determine a safe dose of supplements for you.
- A higher estimate is not proof that more sun exposure is appropriate.
- A screening alert is not a diagnosis, and no alert does not guarantee that the level is
  adequate.

**How to use the result safely**

1. Treat the estimate as educational information and check the units: the model uses
   **nmol/L**.
2. Avoid deliberate sun exposure that causes redness or burning. Follow local UV and
   clinical safety advice, and use appropriate clothing, shade, and sunscreen.
3. Do not start, stop, or change supplements based only on this dashboard.
4. Ask a clinician whether a blood test is appropriate, especially if you have symptoms,
   bone disease, kidney or liver disease, thyroid disease, diabetes, malabsorption risk,
   take medicines affecting vitamin D, are pregnant, or are caring for a child.
5. Use a laboratory 25(OH)D result, when clinically indicated, as the measurement for
   medical decisions. The app can store a user-consented lab result separately from the
   model estimate.

This project was trained using US survey data and has not been clinically validated on
Indian patients. A patient should discuss the result with a qualified healthcare
professional rather than treating the dashboard as a medical device.

### 13.2 Suggested paper structure

The following structure can be used when converting this project description into a paper:

1. **Introduction:** explain vitamin D monitoring, UV exposure, the motivation for a
   low-cost connected prototype, and the distinction between education and diagnosis.
2. **Objectives:** state the hardware, software, prediction, tracking, and safety objectives.
3. **Materials and methods:** describe the ESP32/GUVA-S12SD hardware, calibration, Flask and
   SQLite application, NHANES files, inclusion criteria, feature engineering, model
   hyperparameters, and the nested deficiency evaluation.
4. **Results:** report the measured model metrics, sample sizes, deficiency confusion counts,
   confidence intervals, sensor behaviour, and software verification results. Do not report
   illustrative dashboard values as model validation results.
5. **Discussion:** interpret the model as a population-pattern screening estimate, discuss
   class imbalance and false positives, and explain why the sun model's R² is not directly
   comparable with the basic model on a different population subset.
6. **Safety, ethics, and privacy:** describe consent for lab records, limitations of US
   NHANES data, non-diagnostic use, open device endpoints on a trusted network, and the need
   for clinical confirmation.
7. **Conclusion and future work:** summarise the prototype without claiming clinical
   effectiveness and propose local validation, better UV calibration, security hardening,
   and prospective user studies.

**Recommended wording for the main conclusion**

> The prototype demonstrates an end-to-end method for collecting UV readings, tracking
> exposure sessions, and presenting a NHANES-derived serum 25(OH)D screening estimate.
> Validation showed useful ranking signal but substantial individual prediction error.
> The system is an educational research prototype and requires external, local clinical
> validation before any patient-care use.

### 13.3 Reproducibility and reporting checklist

For a paper, project report, or review by a clinician, record:

- NHANES cycle, source files, download date, and the exact target definition
  (`LBXVIDMS < 30 nmol/L` for deficiency).
- Number of rows before and after each cleaning step, missing-value handling, and valid
  ranges for every feature.
- Model feature lists, hyperparameters, random seeds, software versions, and saved artifact
  names.
- Whether each metric is from saved-artifact verification, cross-validation, or the nested
  outer validation report. These evaluation types should not be mixed.
- Confusion-matrix counts, prevalence, threshold-selection procedure, and precision-recall
  trade-off.
- The date and version of `deficiency_validation_report.json`,
  `health_adjustments.json`, and the model artifacts used in the screenshots.
- Results of `check_model.py`, `check_health_adjustments.py`, dashboard checks, and
  `git diff --check`.

The current verification baseline is 22/22 checks passed. The current nested screening
report is the source for the basic 54.5 nmol/L and sun-habits 51.1 nmol/L operating points.
If data, code, or artifacts change, these results must be regenerated rather than copied
forward.

---

## 14. Thesis-ready code excerpts

The following excerpts are simplified, representative versions of the main computational
steps. They are included for thesis methodology and reproducibility. The complete
implementation and the exact run-specific values remain in the project source files,
model artifacts, and JSON reports.

### 14.1 Target and feature preparation

The target is serum 25(OH)D (`LBXVIDMS`) in nmol/L. Invalid target and BMI rows are removed,
and supplement use is encoded as a binary feature. NHANES special codes for outdoor time are
not treated as minutes.

```python
import numpy as np

BASE_FEATURES = ["age", "female", "bmi", "supplement"]
SUN_FEATURES = BASE_FEATURES + ["outdoor_min", "skin_react"]

def prepare_features(frame):
    data = frame[frame["RIDAGEYR"] >= 18].copy()
    data = data.dropna(subset=["BMXBMI", "LBXVIDMS"])
    data["age"] = data["RIDAGEYR"].astype(float)
    data["female"] = (data["RIAGENDR"] == 2).astype(float)
    data["bmi"] = data["BMXBMI"].astype(float)
    data["supplement"] = (data["DSQTVD"].fillna(0) > 0).astype(float)

    for column in ("DED120", "DED125"):
        data[column] = data[column].where(data[column].between(0, 480))
    data["outdoor_min"] = (5 * data["DED120"] + 2 * data["DED125"]) / 7
    data["skin_react"] = data["DED031"].where(data["DED031"].between(1, 5))
    return data

data = prepare_features(nhanes_frame)
y = data["LBXVIDMS"].astype(float)
X_basic = data[BASE_FEATURES]
X_sun = data.dropna(subset=SUN_FEATURES)[SUN_FEATURES]
```

### 14.2 Gradient Boosting regression

The production model predicts a continuous serum level. Its feature order is stored with the
artifact so the Flask backend uses the same encoding as training.

```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    n_estimators=150,
    max_depth=2,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
)
model.fit(X_basic[BASE_FEATURES], y.loc[X_basic.index])
original_prediction = float(model.predict(profile_frame[BASE_FEATURES])[0])
```

The sun-habits model uses `SUN_FEATURES` and is selected only when its supported age range
and required answers are available.

### 14.3 Deficiency definition and screening metrics

The medical outcome is always based on the measured laboratory value. A prediction cutoff is
a separate screening operating point and does not redefine deficiency.

```python
from sklearn.metrics import precision_score, recall_score, f1_score

def actual_deficient(level_nmol):
    return np.asarray(level_nmol) < 30.0

def score_screening(y_level, predicted_level, cutoff):
    actual = actual_deficient(y_level)
    predicted = np.asarray(predicted_level) < cutoff
    return {
        "actual_deficient": int(actual.sum()),
        "predicted_deficient": int(predicted.sum()),
        "true_positives": int((actual & predicted).sum()),
        "false_positives": int((~actual & predicted).sum()),
        "false_negatives": int((actual & ~predicted).sum()),
        "true_negatives": int((~actual & ~predicted).sum()),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
    }
```

The full evaluator selects thresholds from inner training-fold out-of-fold predictions and
scores them only on untouched outer folds. This produced the current basic cutoff of
54.5 nmol/L and sun-habits cutoff of 51.1 nmol/L.

### 14.4 Separate health-adjustment layer

The original model prediction is preserved before configurable experimental adjustments are
applied. Deductions are positive magnitudes in configuration and are subtracted once.

```python
def apply_adjustments(original, answers, config):
    deductions = config["binary_deductions_nmol"]
    single_cap = float(config["MAX_SINGLE_HEALTH_DEDUCTION_NMOL"])
    total_cap = float(config["MAX_HEALTH_DEDUCTION_NMOL"])
    total = 0.0
    details = []

    for field, magnitude in deductions.items():
        if answers.get(field) == "yes":
            value = min(float(magnitude), single_cap)
            total += value
            details.append({"field": field, "deduction_nmol": value})

    total = min(total, total_cap)
    milk_bonus = float(config["milk_bonus_by_condition"].get(
        answers.get("milk"), 0.0
    ))
    final = np.clip(
        float(original) - total + milk_bonus,
        config["minimum_prediction_nmol"],
        config["maximum_prediction_nmol"],
    )
    return original, details, total, milk_bonus, float(final)
```

The complete backend applies the kidney/diabetes overlap rule before the combined cap.
`No`, missing, and unrecognised binary answers contribute zero. The health values and milk
bonus are academic demonstration heuristics, not clinical effect sizes.

### 14.5 UV index and exposure calculation

The ESP32 converts the sensor voltage to UV index using the calibrated project constants.
Saturation is flagged because the sensor cannot measure beyond its effective range.

```python
BASE_MV = 142.0
MV_PER_UVI = 375.0
MAX_UVI = 7.0

def millivolts_to_uvi(millivolts):
    if millivolts >= 3000:
        return MAX_UVI, True
    uvi = max(0.0, (float(millivolts) - BASE_MV) / MV_PER_UVI)
    return min(uvi, MAX_UVI), False

def dose_from_reading(uvi, seconds):
    elapsed = min(max(float(seconds), 0.0), 15.0)
    return float(uvi) * 0.025 * elapsed
```

The UV-dose calculation is a planning approximation and does not measure blood vitamin D.
A laboratory 25(OH)D test is required for clinical measurement.

### 14.6 Verification commands

```text
python check_model.py
python check_health_adjustments.py
python check_deficiency_dashboard.py
python check_progress_baseline.py
git diff --check
```

The current saved baseline reports 22/22 model checks passed. Exact screening metrics should
always be copied from `deficiency_validation_report.json` after the evaluation run.

---

## 15. Future work

1. **Security:** device key for the ESP32, CSRF protection, HTTPS, rate limits.
2. **Wireless and power:** battery or power-bank operation (always-on power bank), enclosure with a diffuser; Bluetooth only if WiFi is not available.
3. **Hosting:** run the server on a Raspberry Pi or a cloud service so the laptop need not stay on.
4. **Better sensing:** a sensor with a wider range (for example a digital UV sensor), proper calibration against a reference meter.
5. **Local data:** collect lab results from consenting users (the `/labs` table already stores them with the model's prediction and the profile at that time) and use them to check or retrain the model for India.
6. **Model improvements:** monotonic constraint for skin reaction, more features (diet, latitude, season), simple uncertainty estimates, a model for ages below 20 and above 59 with outdoor-time data.
7. **Features:** reminders and notifications, weekly reports and export, multi-language interface, mobile-friendly app.
8. **Testing:** automated tests for the exposure maths and API, and a user study.

---

## 16. How to run

1. **ESP32:** save `ssd1306.py` and `main.py` on the board (Thonny), with your WiFi name, password and the laptop IP in `SERVER_URL`. Use a 2.4 GHz network.
2. **Laptop (VS Code terminal, inside the project folder with the virtual environment active):**
   - `pip install flask pandas numpy scikit-learn joblib matplotlib`
   - `python download_data.py` (downloads NHANES files into `data/`)
   - Run the training scripts so `vitd_model.joblib` and `vitd_model_sun.joblib` appear.
   - `python server.py`, then open `http://localhost:5000`.
3. Register, fill in the profile, and watch the dashboard; the OLED shows the same numbers.
4. For reports: `python check_model.py`, `python check_metrics.py`, `python make_charts.py`.

---

## 17. Summary (for the abstract)

A low-cost IoT system built on an ESP32 and a GUVA-S12SD analog UV sensor measures the UV
index, shows it on an OLED display, and streams it over WiFi to a Flask/SQLite web
application that supports multiple users. Each user's profile (age, sex, BMI, skin type from
a 5-question quiz, clothing, sunscreen, supplement use and time outdoors) feeds two
Gradient Boosting regression models trained on the US NHANES 2013-2014 dataset to estimate
serum 25(OH)D (MAE about 17-18 nmol/L, R² 0.19-0.26). A separate nested validation study
reports ROC-AUC 0.718 for the basic screening model and 0.727 for the sun-habits model at
the medical outcome `LBXVIDMS < 30 nmol/L`. The predicted gap to a 75 nmol/L planning target
is used for educational goal-setting, while a dose-based exposure model estimates burn risk
and approximate session contribution. The dashboard and OLED report progress live. Health
deductions and milk bonuses are configurable academic heuristics, not clinical effect sizes.
The project is an educational prototype; the models are not validated for Indian users or
patient care, and further work includes local data collection, better sensing, security,
and external clinical validation.
