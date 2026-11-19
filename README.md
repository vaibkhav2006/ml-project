# Weather-Based Umbrella Predictor

An AI/ML project that predicts whether it will rain, so you know whether to carry an umbrella.

## Problem
People often forget to carry an umbrella when weather conditions indicate rain.

## Approach
- Small weather dataset with **humidity, cloud cover, temperature and rainfall**
- Binary classifier (Logistic Regression) predicts **Rain / No Rain**
- A simple real-life classification example using weather features

## How to run
```bash
pip install -r requirements.txt
python umbrella_predictor.py
```
The script creates `weather_data.csv` on the first run, trains the model, prints accuracy,
and then lets you enter weather values to get an umbrella recommendation.

## Author
Vaibhav Kumar - 124183 - Batch B1
