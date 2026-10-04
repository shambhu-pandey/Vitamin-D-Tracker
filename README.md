# Vitamin-D-Tracker

This project measures live UV index and converts it into **personal vitamin D guidance**.  
Many people do not know how long they should stay in sunlight to make enough vitamin D, or when UV exposure becomes a burn risk. The answer depends on both real-time UV conditions and personal factors.

## Overview

The system combines hardware sensing, server-side calculations, and machine learning to provide user-specific recommendations.

### 1) Device (ESP32 + UV sensor + OLED)

- Reads UV values every few seconds
- Displays current UV index on an OLED screen
- Sends readings to the server over WiFi

### 2) Server (Flask + SQLite)

- Stores user accounts, profile factors, and UV readings
- Runs sun-exposure and vitamin D production calculations
- Runs the trained ML model for vitamin D estimation
- Serves the web dashboard

### 3) Machine Learning Model (trained on NHANES)

- Predicts blood vitamin D level (25-hydroxy vitamin D, nmol/L) from profile inputs
- Uses the gap to a healthy target to estimate each user’s daily vitamin D goal

## What the dashboard provides

For each user, the dashboard reports:

- Current UV index
- Vitamin D generated so far today
- Remaining sun-exposure minutes needed to meet the daily goal
- Remaining minutes before reaching estimated burn risk
