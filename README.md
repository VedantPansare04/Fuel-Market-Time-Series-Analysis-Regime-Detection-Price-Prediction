# Fuel Market Stress and Forecasting

This project analyzes historical fuel-market data with a mix of exploratory time-series analysis, regime-change detection, and forecasting. The main upgrade beyond a standard market notebook is the use of Jensen-Shannon divergence to detect when recent return distributions diverge from a long-run baseline.

## Project Goals

- Compare long-run price behavior across major fuel markets.
- Measure volatility and drawdowns to understand stress periods.
- Study cross-market dependence through static and rolling correlations.
- Detect structural regime changes using Jensen-Shannon divergence.
- Benchmark next-day Crude Oil return forecasting with a simple Ridge model.

## Dataset

The repository contains six CSV files in the `Datasets/` folder:

- `all_fuels_data.csv`
- `Brent_Crude_Oil_data.csv`
- `Crude_Oil_data.csv`
- `Heating_Oil_data.csv`
- `Natural_Gas_data.csv`
- `RBOB_Gasoline_data.csv`

The combined dataset covers five commodities from 2000-08-23 through 2024-06-24, with Brent Crude Oil beginning in 2007.

## Repository Structure

```text
fuel-market-stress-and-forecasting/
|-- Datasets/
|-- Codes/
|   |-- fuel_market_analysis.py
|   |-- fuel_market_stress_project.ipynb
|   `-- requirements.txt
|-- Outputs/
|   |-- figures/
|   |-- tables/
|   `-- observations_and_conclusion.md
|-- resume_bullets.md
`-- README.md
```

## Methods Used

1. Loaded and cleaned the combined fuel-market dataset with `pandas`.
2. Computed daily returns, rolling annualized volatility, and drawdowns.
3. Built normalized price series to compare markets on a common base.
4. Calculated correlation matrices and 90-day rolling correlations.
5. Applied rolling Jensen-Shannon divergence to flag regime changes in return distributions.
6. Built a time-based forecasting benchmark for next-day Crude Oil returns using lagged multi-market features and Ridge regression.

## Key Findings

- Crude Oil is the most volatile market in the sample.
- Natural Gas is the least synchronized with the petroleum-linked contracts.
- Divergence spikes reveal market regime shifts that are not obvious from price levels alone.
- The Ridge model beats a naive lag baseline, but forecasting next-day returns remains difficult.
- The 2020 negative Crude Oil print is a major anomaly and should be handled carefully in risk analytics.

## Outputs Included

- Saved figures for normalized prices, volatility, drawdown, correlations, divergence, and forecast comparison.
- Saved tables for market summaries, drawdowns, correlations, divergence peaks, and forecasting metrics.
- A written findings note in `Outputs/observations_and_conclusion.md`.

## How to Run

```bash
pip install -r Codes/requirements.txt
python Codes/fuel_market_analysis.py
```

Running the script will regenerate the figures and tables in `Outputs/`.

## Tools and Libraries

- Python
- NumPy
- pandas
- matplotlib
- SciPy
- scikit-learn

