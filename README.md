# Vitamin-D-Tracker

## 1. Overview

This project measures the live UV index with a low-cost sensor and turns it into **personal vitamin D advice**. Many people do not know how long they should stay in the sun to make enough vitamin D, or when the sun becomes a burn risk. The answer depends on the person (skin type, age, clothing, supplements, body weight) and on the UV level right now.

The system has three parts:

1. **Device (ESP32 + sensor + OLED):** reads UV every few seconds, shows it on a small screen, and sends it over WiFi to the server.
2. **Server (Flask + SQLite):** stores users, profiles and readings, runs the sun-exposure calculations, runs the trained ML model, and serves the dashboard.
3. **Machine-learning model (trained on NHANES):** predicts a person's blood vitamin D level (25-hydroxy vitamin D, in nmol/L) from their profile. The gap to a healthy target decides the daily vitamin D goal.

The dashboard then tells each user: how much UV there is now, how much vitamin D they have made today, how many more minutes they need in the sun, and how many minutes are left before the burn limit.
