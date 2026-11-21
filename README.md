# Predictive Analytics Using Historical Data

## Project question

Can monthly international airline passenger totals be forecast from their historical trend and recurring yearly seasonality?

## Included files

- `air_passengers.csv` — 144 monthly observations from January 1949 through December 1960. Counts are in thousands.
- `forecast.py` — data preparation, chronological 80/20 validation, Prophet modeling, MAE/RMSE/MAPE evaluation, and a final 12-month forecast.
- Running the script creates the holdout and future forecast CSVs and PNG charts in this folder.

## Run it

Use Python 3.9 or newer. From this folder:

```bash
python -m pip install pandas numpy matplotlib prophet
python forecast.py
```

## Method

The latest 20% of months are held out in chronological order, and the model is fit only on the earlier 80%. This avoids training on future observations. Error metrics compare the forecast with the known held-out values. Once validation is complete, a fresh model is fit on the full series and generates the following 12 monthly predictions. Prophet decomposes the signal into trend and yearly seasonality and also returns uncertainty intervals.

## Interpretation limits

This is a small, familiar demonstration series, not a current airline planning model. The holdout results estimate performance on this historical tail only. Forecasts assume the old trend and seasonal pattern remain informative; structural changes, shocks, and operational drivers are not represented. Treat intervals and point forecasts as estimates, not guarantees.
