# Predictive Analytics Using Historical Data

## Objective

Demonstrate an end-to-end forecasting workflow by using historical monthly airline passenger totals to estimate future demand.

## Data and preparation

The supplied CSV contains 144 monthly observations from January 1949 to December 1960, measured in thousands of passengers. The script parses the month field, converts counts to numeric values, removes rows with invalid date/count fields, sorts observations by date, and rejects duplicate months. The resulting series shows an upward long-term trend and a recurring annual pattern.

## Modeling and validation

The project uses Prophet with yearly seasonality. The series is split chronologically: the first 80% is training data and the final 20% is a holdout set (115 training months and 29 test months). The validation model forecasts exactly the holdout months. On this test period, MAE was **33.90 thousand passengers**, RMSE was **41.33 thousand passengers**, and MAPE was **7.72%**. The script writes individual predictions to `holdout_predictions.csv`; `holdout_forecast.png` compares the forecast with training history and actual holdout observations.

After evaluation, a separate Prophet model is trained on all 144 observations and forecasts January through December 1961. The predicted monthly values range from about **459.3 to 577.1 thousand passengers**. The point forecasts and lower/upper uncertainty bounds are written to `next_12_month_forecast.csv`, with a chart saved as `next_12_month_forecast.png`.

## Outcome and conclusion

The resulting forecast captures the series' upward trend and annual seasonality, with a time-aware evaluation. The test error is about 7.72% by MAPE on this historical holdout. Before using this result for a real business decision, compare it with simple baselines and newer data, inspect residuals, and account for changes that the historical series cannot explain.

## Reproduction

Install the packages listed in `README.md` and run `python forecast.py` from this folder. The code, input data, and generated outputs stay together so the analysis can be rerun and reviewed.
