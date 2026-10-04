# Personalised UV Index and Vitamin D Tracker

**An ESP32 + UV sensor + Flask web app + machine-learning project**

Dataset: US NHANES 2013-2014 | Hardware: ESP32, GUVA-S12SD, SSD1306 OLED | Software: MicroPython, Python, Flask, SQLite, scikit-learn

---

## 1. Overview

This project measures the live UV index with a low-cost sensor and turns it into **personal vitamin D advice**. Many people do not know how long they should stay in the sun to make enough vitamin D, or when the sun becomes a burn risk. The answer depends on the person (skin type, age, clothing, supplements, body weight) and on the UV level right now.

The system has three parts:

1. **Device (ESP32 + sensor + OLED):** reads UV every few seconds, shows it on a small screen, and sends it over WiFi to the server.
2. **Server (Flask + SQLite):** stores users, profiles and readings, runs the sun-exposure calculations, runs the trained ML model, and serves the dashboard.
3. **Machine-learning model (trained on NHANES):** predicts a person's blood vitamin D level (25-hydroxy vitamin D, in nmol/L) from their profile. The gap to a healthy target decides the daily vitamin D goal.

The dashboard then tells each user: how much UV there is now, how much vitamin D they have made today, how many more minutes they need in the sun, and how many minutes are left before the burn limit.

> **Important:** this is an educational project. It gives estimates only. It is not a medical device and not a blood test.

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
- **Weakness:** at the 50 nmol/L cut-off the models catch only about 10% (basic) and 26% (sun) of truly low people, because predictions are pulled towards the average. The models never predict "deficient" (below 30). They are good at **ranking** risk (AUC about 0.72-0.76), not at diagnosing.

### 9.6 From prediction to the daily goal

1. Predicted level (nmol/L) from the model.
2. **Gap** = 75 - predicted level (target 75 nmol/L, never below 0).
3. **IU to close the gap:** about 2 nmol/L rise per 100 IU a day (from published trials), so 1 nmol/L is about 50 IU/day; capped at 2000 IU.
4. **Daily goal** = the larger of the model's gap in IU and the standard recommended intake (600 IU, 800 IU above age 70). The goal is never set below the standard intake because the model error is large.

### 9.7 How the result is shown (honest wording)

- Label: "Predicted level: LOW range (below 50) / MIDDLE range (50-75) / HIGHER range (75+)", plus a colour scale with the likely range (+/- typical error).
- **"Higher risk of low vitamin D" flag** when the predicted level is below 62 (sun model) or 64 (basic model). Cut-offs were chosen from the cross-validated results: they find about 80% of people truly below 50 nmol/L, with about half of flagged people actually fine. Not being flagged does not mean the level is good.
- A reminder that predictions are pulled towards the average and that only a blood test can confirm.

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

---

## 14. Future work

1. **Security:** device key for the ESP32, CSRF protection, HTTPS, rate limits.
2. **Wireless and power:** battery or power-bank operation (always-on power bank), enclosure with a diffuser; Bluetooth only if WiFi is not available.
3. **Hosting:** run the server on a Raspberry Pi or a cloud service so the laptop need not stay on.
4. **Better sensing:** a sensor with a wider range (for example a digital UV sensor), proper calibration against a reference meter.
5. **Local data:** collect lab results from consenting users (the `/labs` table already stores them with the model's prediction and the profile at that time) and use them to check or retrain the model for India.
6. **Model improvements:** monotonic constraint for skin reaction, more features (diet, latitude, season), simple uncertainty estimates, a model for ages below 20 and above 59 with outdoor-time data.
7. **Features:** reminders and notifications, weekly reports and export, multi-language interface, mobile-friendly app.
8. **Testing:** automated tests for the exposure maths and API, and a user study.

---

## 15. How to run

1. **ESP32:** save `ssd1306.py` and `main.py` on the board (Thonny), with your WiFi name, password and the laptop IP in `SERVER_URL`. Use a 2.4 GHz network.
2. **Laptop (VS Code terminal, inside the project folder with the virtual environment active):**
   - `pip install flask pandas numpy scikit-learn joblib matplotlib`
   - `python download_data.py` (downloads NHANES files into `data/`)
   - Run the training scripts so `vitd_model.joblib` and `vitd_model_sun.joblib` appear.
   - `python server.py`, then open `http://localhost:5000`.
3. Register, fill in the profile, and watch the dashboard; the OLED shows the same numbers.
4. For reports: `python check_model.py`, `python check_metrics.py`, `python make_charts.py`.

---

## 16. Summary (for the abstract)

A low-cost IoT system built on an ESP32 and a GUVA-S12SD analog UV sensor measures the UV index, shows it on an OLED display, and streams it over WiFi to a Flask/SQLite web application that supports multiple users. Each user's profile (age, sex, BMI, skin type from a 5-question quiz, clothing, sunscreen, supplement use and time outdoors) feeds two gradient-boosting regression models trained on the US NHANES 2013-2014 dataset to estimate blood vitamin D level (MAE about 17-18 nmol/L, R2 0.19-0.25, ROC AUC about 0.72-0.76). The predicted gap to a 75 nmol/L target sets a personal daily vitamin D goal, while a dose-based exposure model gives the burn limit and the minutes of sun needed at the current UV. The dashboard and OLED report progress live. The project is an educational prototype; the models are not validated for Indian users, and further work includes local data collection, better sensing, security and hosting.
