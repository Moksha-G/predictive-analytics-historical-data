"""Monthly Air Passengers forecasting project using Prophet."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "air_passengers.csv"


def main():
    # Load and clean: parse monthly dates, coerce passenger counts, sort, and
    # remove incomplete records. Duplicate months are rejected rather than
    # silently aggregated.
    raw = pd.read_csv(DATA_PATH)
    raw["Month"] = pd.to_datetime(raw["Month"], errors="coerce")
    raw["Passengers"] = pd.to_numeric(raw["Passengers"], errors="coerce")
    clean = raw.dropna(subset=["Month", "Passengers"]).sort_values("Month")
    if clean["Month"].duplicated().any():
        raise ValueError("Dataset contains duplicate months.")
    if len(clean) < 24:
        raise ValueError("At least 24 monthly observations are required.")

    model_data = clean.rename(columns={"Month": "ds", "Passengers": "y"})
    split = int(len(model_data) * 0.8)
    train, test = model_data.iloc[:split], model_data.iloc[split:]
    print(f"Observations: {len(model_data)} | Train: {len(train)} | Test: {len(test)}")
    print(f"Train through {train.ds.max():%Y-%m}; test {test.ds.min():%Y-%m} to {test.ds.max():%Y-%m}")

    # Keep the latest 20% as a contiguous holdout, as required for forecasting.
    validation_model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                               daily_seasonality=False)
    validation_model.fit(train)
    validation_forecast = validation_model.predict(test[["ds"]])
    actual = test["y"].to_numpy()
    predicted = validation_forecast["yhat"].to_numpy()
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    print(f"Holdout MAE: {mae:.2f} thousand passengers")
    print(f"Holdout RMSE: {rmse:.2f} thousand passengers")
    print(f"Holdout MAPE: {mape:.2f}%")

    pd.DataFrame({"Month": test.ds, "Actual": actual, "Predicted": predicted,
                  "Error": actual - predicted}).to_csv(ROOT / "holdout_predictions.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(train.ds, train.y, color="#4263eb", label="Training history")
    ax.plot(test.ds, actual, color="#172b4d", label="Actual holdout")
    ax.plot(test.ds, predicted, color="#e8590c", linestyle="--", label="Forecast")
    ax.fill_between(test.ds, validation_forecast.yhat_lower,
                    validation_forecast.yhat_upper, color="#e8590c", alpha=.15,
                    label="Uncertainty interval")
    ax.set(title="Air Passengers: holdout forecast", ylabel="Passengers (thousands)", xlabel="Month")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(ROOT / "holdout_forecast.png", dpi=180)
    plt.close(fig)

    # After honest holdout evaluation, refit on all available observations to
    # produce the requested operational forecast for the next 12 months.
    final_model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                          daily_seasonality=False)
    final_model.fit(model_data)
    future = final_model.make_future_dataframe(periods=12, freq="MS")
    future_forecast = final_model.predict(future)
    next_year = future_forecast.tail(12)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    next_year.to_csv(ROOT / "next_12_month_forecast.csv", index=False)
    fig = final_model.plot(future_forecast)
    fig.set_size_inches(11, 5.5)
    plt.title("Air Passengers: next 12 months forecast")
    plt.ylabel("Passengers (thousands)")
    plt.xlabel("Month")
    plt.tight_layout()
    fig.savefig(ROOT / "next_12_month_forecast.png", dpi=180)
    plt.close(fig)
    print("Saved holdout and 12-month forecast tables and charts in", ROOT)
    print(next_year.to_string(index=False))


if __name__ == "__main__":
    main()
