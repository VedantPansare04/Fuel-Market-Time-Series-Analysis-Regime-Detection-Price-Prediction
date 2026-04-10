from pathlib import Path
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
plt.style.use("ggplot")
pd.options.display.float_format = "{:,.4f}".format
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Datasets"
OUTPUT_DIR = ROOT / "Outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"


def resolve_data_path() -> Path:
    candidate = DATA_DIR / "all_fuels_data.csv"
    if candidate.exists():
        return candidate
    raise FileNotFoundError("Could not find Datasets/all_fuels_data.csv.")


def load_market_data(path: Path) -> pd.DataFrame:
    fuel = pd.read_csv(path, parse_dates=["date"]).sort_values(["commodity", "date"]).copy()
    fuel["daily_return"] = fuel.groupby("commodity")["close"].pct_change(fill_method=None)
    fuel["volume_change"] = fuel.groupby("commodity")["volume"].pct_change(fill_method=None)
    fuel["rolling_vol_21"] = (
        fuel.groupby("commodity")["daily_return"]
        .transform(lambda series: series.rolling(21).std() * np.sqrt(252))
    )
    fuel["drawdown"] = fuel.groupby("commodity")["close"].transform(
        lambda series: series / series.cummax() - 1
    )
    return fuel


def build_normalized_index(fuel: pd.DataFrame) -> pd.DataFrame:
    wide_close = fuel.pivot(index="date", columns="commodity", values="close").sort_index()
    return wide_close.apply(
        lambda series: (series / series.dropna().iloc[0]) * 100 if series.notna().any() else series
    )


def rolling_js_divergence(
    fuel: pd.DataFrame,
    window: int = 126,
    baseline_window: int = 756,
    bins: np.ndarray | None = None,
) -> pd.DataFrame:
    if bins is None:
        bins = np.linspace(-0.30, 0.30, 61)

    records = []
    lower = bins[0] + 1e-9
    upper = bins[-1] - 1e-9

    for commodity, group in fuel.groupby("commodity"):
        data = group.dropna(subset=["daily_return"]).copy()
        values = data["daily_return"].clip(lower, upper).to_numpy()

        if len(values) < baseline_window + window:
            continue

        baseline = values[:baseline_window]
        baseline_hist, _ = np.histogram(baseline, bins=bins, density=True)
        baseline_hist = np.where(baseline_hist <= 0, 1e-12, baseline_hist)
        baseline_hist = baseline_hist / baseline_hist.sum()

        for end in range(window, len(values) + 1):
            current = values[end - window : end]
            current_hist, _ = np.histogram(current, bins=bins, density=True)
            current_hist = np.where(current_hist <= 0, 1e-12, current_hist)
            current_hist = current_hist / current_hist.sum()

            records.append(
                {
                    "commodity": commodity,
                    "date": data.iloc[end - 1]["date"],
                    "js_divergence": float(
                        jensenshannon(current_hist, baseline_hist, base=2.0) ** 2
                    ),
                }
            )

    return pd.DataFrame(records)


def build_forecast_frame(
    fuel: pd.DataFrame,
    target_commodity: str = "Crude Oil",
) -> pd.DataFrame:
    close_wide = fuel.pivot(index="date", columns="commodity", values="close").sort_index()
    volume_wide = fuel.pivot(index="date", columns="commodity", values="volume").sort_index()

    returns = (
        close_wide.pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .clip(-0.25, 0.25)
    )
    volume_change = (
        volume_wide.pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .clip(-1.0, 1.0)
    )

    frame = pd.DataFrame(index=close_wide.index)

    for commodity in close_wide.columns:
        safe_name = commodity.lower().replace(" ", "_")
        frame[f"{safe_name}_ret_lag_1"] = returns[commodity].shift(1)
        frame[f"{safe_name}_ret_lag_2"] = returns[commodity].shift(2)
        frame[f"{safe_name}_ret_lag_5"] = returns[commodity].rolling(5).mean().shift(1)
        frame[f"{safe_name}_vol_21"] = returns[commodity].rolling(21).std().shift(1)
        frame[f"{safe_name}_volchg_1"] = volume_change[commodity].shift(1)

    frame["target_return"] = returns[target_commodity].shift(-1)
    return frame.dropna()


def save_normalized_prices(normalized_index: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    for column in normalized_index.columns:
        ax.plot(normalized_index.index, normalized_index[column], linewidth=2, label=column)

    ax.set_title("Normalized Fuel Prices (Base = 100 at Each Series Start)")
    ax.set_ylabel("Indexed Close Price")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "normalized_fuel_prices.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_volatility_and_drawdown(fuel: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for commodity, group in fuel.groupby("commodity"):
        axes[0].plot(group["date"], group["rolling_vol_21"], linewidth=1.7, label=commodity)
        axes[1].plot(group["date"], group["drawdown"], linewidth=1.7, label=commodity)

    axes[0].set_title("21-Day Rolling Volatility (Annualized)")
    axes[0].set_ylabel("Volatility")
    axes[0].legend(loc="upper left", ncol=2)

    axes[1].set_title("Drawdown from Running Peak")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "volatility_and_drawdown.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmap(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")

    ax.set_title("Correlation of Daily Returns")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_rolling_correlation(rolling_corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    for column in rolling_corr.columns:
        ax.plot(rolling_corr.index, rolling_corr[column], linewidth=1.8, label=column)

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("90-Day Rolling Correlation with Crude Oil")
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Date")
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rolling_correlation_with_crude.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_js_divergence(stress: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    for commodity, group in stress.groupby("commodity"):
        ax.plot(group["date"], group["js_divergence"], linewidth=1.8, label=commodity)

    ax.set_title("Rolling Jensen-Shannon Divergence of Return Distributions")
    ax.set_ylabel("JS Divergence")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rolling_js_divergence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_forecast_plot(forecast_results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    forecast_results.iloc[:250].plot(ax=ax, linewidth=1.4)
    ax.set_title("Next-Day Crude Oil Return Forecasts on the Test Window")
    ax.set_ylabel("Return")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "forecast_vs_actual.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_tables(results: dict[str, pd.DataFrame | pd.Series]) -> None:
    for name, frame in results.items():
        if isinstance(frame, pd.Series):
            frame = frame.to_frame()
        frame.to_csv(TABLES_DIR / f"{name}.csv")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    fuel = load_market_data(resolve_data_path())

    project_summary = (
        fuel.groupby("commodity")
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            observations=("date", "size"),
            avg_close=("close", "mean"),
            avg_volume=("volume", "mean"),
        )
        .sort_values("start_date")
    )

    quality_check = (
        fuel.groupby("commodity")
        .agg(
            missing_close=("close", lambda series: int(series.isna().sum())),
            min_close=("close", "min"),
            max_close=("close", "max"),
            min_return=("daily_return", "min"),
            max_return=("daily_return", "max"),
        )
        .sort_index()
    )

    negative_close_rows = fuel.loc[
        fuel["close"] <= 0, ["commodity", "date", "close", "volume"]
    ].copy()
    deepest_drawdowns = (
        fuel.groupby("commodity")["drawdown"]
        .min()
        .sort_values()
        .rename("deepest_drawdown")
    )
    annualized_volatility = (
        fuel.groupby("commodity")["daily_return"].std().mul(np.sqrt(252)).sort_values(ascending=False)
    )
    latest_drawdown = (
        fuel.sort_values("date")
        .groupby("commodity")
        .tail(1)
        .set_index("commodity")["drawdown"]
        .sort_values()
    )

    normalized_index = build_normalized_index(fuel)
    return_wide = fuel.pivot(index="date", columns="commodity", values="daily_return").sort_index()
    corr = return_wide.corr()
    rolling_corr = pd.DataFrame(
        {
            commodity: return_wide["Crude Oil"].rolling(90).corr(return_wide[commodity])
            for commodity in return_wide.columns
            if commodity != "Crude Oil"
        }
    )
    rolling_corr_range = rolling_corr.agg(["min", "max"])

    stress = rolling_js_divergence(fuel, window=126, baseline_window=756)
    top_stress_dates = (
        stress.sort_values("js_divergence", ascending=False)
        .groupby("commodity")
        .head(3)
        .sort_values(["commodity", "js_divergence"], ascending=[True, False])
        .reset_index(drop=True)
    )
    peak_js = (
        stress.sort_values("js_divergence", ascending=False)
        .groupby("commodity")
        .head(1)
        .sort_values("js_divergence", ascending=False)
        .reset_index(drop=True)
    )

    forecast_frame = build_forecast_frame(fuel, target_commodity="Crude Oil")
    split_idx = int(len(forecast_frame) * 0.8)
    train = forecast_frame.iloc[:split_idx]
    test = forecast_frame.iloc[split_idx:]

    X_train = train.drop(columns="target_return")
    y_train = train["target_return"]
    X_test = test.drop(columns="target_return")
    y_test = test["target_return"]

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    ridge_pred = pd.Series(ridge.predict(X_test), index=X_test.index, name="ridge_prediction")
    naive_pred = X_test["crude_oil_ret_lag_1"].rename("naive_prediction")

    metrics = (
        pd.DataFrame(
            [
                {
                    "model": "Ridge",
                    "MAE": mean_absolute_error(y_test, ridge_pred),
                    "RMSE": mean_squared_error(y_test, ridge_pred) ** 0.5,
                    "R2": r2_score(y_test, ridge_pred),
                    "direction_accuracy": (np.sign(ridge_pred) == np.sign(y_test)).mean(),
                },
                {
                    "model": "Naive last return",
                    "MAE": mean_absolute_error(y_test, naive_pred),
                    "RMSE": mean_squared_error(y_test, naive_pred) ** 0.5,
                    "R2": r2_score(y_test, naive_pred),
                    "direction_accuracy": (np.sign(naive_pred) == np.sign(y_test)).mean(),
                },
            ]
        )
        .set_index("model")
        .sort_index()
    )

    top_coefficients = (
        pd.Series(ridge.coef_, index=X_train.columns)
        .sort_values(key=np.abs, ascending=False)
        .head(12)
        .rename("coefficient")
    )

    forecast_results = pd.concat(
        [y_test.rename("actual_return"), ridge_pred, naive_pred],
        axis=1,
    )

    save_normalized_prices(normalized_index)
    save_volatility_and_drawdown(fuel)
    save_correlation_heatmap(corr)
    save_rolling_correlation(rolling_corr)
    save_js_divergence(stress)
    save_forecast_plot(forecast_results)

    save_tables(
        {
            "project_summary": project_summary,
            "quality_check": quality_check,
            "negative_close_rows": negative_close_rows,
            "deepest_drawdowns": deepest_drawdowns,
            "annualized_volatility": annualized_volatility,
            "latest_drawdown": latest_drawdown,
            "correlation_matrix": corr,
            "rolling_correlation_range": rolling_corr_range,
            "top_stress_dates": top_stress_dates,
            "peak_js_divergence": peak_js,
            "forecast_metrics": metrics,
            "top_ridge_coefficients": top_coefficients,
            "forecast_preview": forecast_results.head(50),
        }
    )

    print(f"Saved figures to: {FIGURES_DIR}")
    print(f"Saved tables to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
