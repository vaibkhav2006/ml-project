"""
Weather-Based Umbrella Predictor
--------------------------------
Binary classification: predict Rain (1) / No Rain (0) from weather features
(humidity, cloud cover, temperature, rainfall) and tell the user whether to
carry an umbrella.
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_FILE = "weather_data.csv"
FEATURES = ["humidity", "cloud_cover", "temperature", "rainfall"]


def create_dataset(n_samples=500, seed=42):
    """Create a small synthetic weather dataset and save it as a CSV file."""
    rng = np.random.default_rng(seed)

    humidity = rng.uniform(20, 100, n_samples)        # %
    cloud_cover = rng.uniform(0, 100, n_samples)      # %
    temperature = rng.uniform(10, 40, n_samples)      # degree C
    rainfall = rng.exponential(2.0, n_samples)        # mm in last few hours

    # Rain is more likely with high humidity, heavy clouds, cooler temp, recent rain
    score = (
        0.06 * (humidity - 65)
        + 0.05 * (cloud_cover - 55)
        - 0.10 * (temperature - 27)
        + 0.60 * (rainfall - 2)
        + rng.normal(0, 1.0, n_samples)               # noise
    )
    prob = 1 / (1 + np.exp(-score))
    rain = (rng.uniform(0, 1, n_samples) < prob).astype(int)

    df = pd.DataFrame(
        {
            "humidity": humidity.round(1),
            "cloud_cover": cloud_cover.round(1),
            "temperature": temperature.round(1),
            "rainfall": rainfall.round(2),
            "rain": rain,
        }
    )
    df.to_csv(DATA_FILE, index=False)
    return df


def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    print(f"{DATA_FILE} not found - creating a new dataset...")
    return create_dataset()


def train_model(df):
    X = df[FEATURES]
    y = df["rain"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Rain", "Rain"]))
    return model


def predict_umbrella(model, humidity, cloud_cover, temperature, rainfall):
    sample = pd.DataFrame(
        [[humidity, cloud_cover, temperature, rainfall]], columns=FEATURES
    )
    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][1]
    if pred == 1:
        return f"Rain expected ({prob:.0%} chance). Carry an umbrella!"
    return f"No rain expected ({prob:.0%} chance of rain). No umbrella needed."


def main():
    df = load_data()
    print("Dataset preview:")
    print(df.head())
    print(f"\nTotal rows: {len(df)} | Rain days: {df['rain'].sum()}")

    model = train_model(df)

    print("\n=== Umbrella Predictor ===")
    print("Enter today's weather (or type 'q' to quit).")
    while True:
        try:
            h = input("\nHumidity (%): ")
            if h.lower() == "q":
                break
            c = input("Cloud cover (%): ")
            t = input("Temperature (C): ")
            r = input("Rainfall (mm): ")
            print(predict_umbrella(model, float(h), float(c), float(t), float(r)))
        except ValueError:
            print("Please enter numbers only.")
        except (KeyboardInterrupt, EOFError):
            break
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
