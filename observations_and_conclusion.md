# Observations and Conclusion

## Observations

1. Crude Oil is the most volatile series in the dataset, with annualized volatility of about 79.7%, while Natural Gas is the second most volatile at about 59.0%.
2. The dataset contains a real market anomaly on 2020-04-20, when Crude Oil closed at -37.63. This drives the deepest crude drawdown below -100% and is an important data-quality note for any financial modeling discussion.
3. Brent Crude Oil, Heating Oil, and RBOB Gasoline show strong positive co-movement. Their daily return correlations range from about 0.71 to 0.81, while Natural Gas behaves much more independently.
4. Jensen-Shannon divergence highlights regime shifts that ordinary price charts do not show clearly. The strongest stress periods appear around March 2009 for Crude Oil, August 2014 for Brent, Heating Oil, and RBOB Gasoline, and August 2018 for Natural Gas.
5. Natural Gas ends the sample in the weakest state, with a latest drawdown of about -81.7%, much worse than the other fuels.
6. The Ridge forecasting benchmark improves on a naive lag-based baseline. It reduces MAE from 0.0274 to 0.0189 and improves direction accuracy from 45.3% to 49.1%, but the negative R2 shows that short-horizon return prediction remains difficult.

## Conclusion

This project shows that statistical divergence can add useful structure to time-series market analysis. Instead of looking only at price direction, Jensen-Shannon divergence measures when the return distribution itself changes, which helps identify stress regimes and breaks in market behavior. In this dataset, petroleum-linked markets remain tightly connected, while Natural Gas follows a more distinct path.

The forecasting experiment also reinforces a realistic conclusion: even with cross-market lagged features and volume changes, next-day return prediction is still noisy. That makes the project stronger from a portfolio perspective because it does not force an unrealistic success story. The main value here is the combination of data cleaning, exploratory analysis, divergence-based regime detection, and honest benchmarking.

