import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


class ModelInfo:
    def __init__(
        self,
        name: str,
        train_mse: np.ndarray,
        val_mse: np.ndarray,
        predictions: np.ndarray = np.array([]),
    ):
        self.name = name
        self.train_mse = train_mse
        # for fix early stopping plot length
        if len(self.train_mse) < 100:
            self.train_mse = np.pad(
                self.train_mse, (0, 100 - len(self.train_mse)), "edge"
            )
        self.val_mse = val_mse
        if len(self.val_mse) < 100:
            self.val_mse = np.pad(self.val_mse, (0, 100 - len(self.val_mse)), "edge")

        self.predictions = predictions


def line_chart(y_vals: list[list], labels: list[str]) -> None:
    fig, ax = plt.subplots()
    ax.figure.set_size_inches(8, 4)

    for idx, y_val in enumerate(y_vals):
        ax.plot(y_val, label=f"{labels[idx]}")
    # ax.set_xlabel("Epochs")
    # ax.set_ylabel("Loss")
    # ax.set_title("ARIMA Loss Metrics")
    ax.legend()
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)


def plot_forecast_with_confidence(
    df,
    predictions,
    all_predictions=None,
    history_days=150,
    confidence_level=0.90,
    figsize=(16, 8),
):
    historical_close = df["close"].tail(history_days).values
    historical_index = np.arange(-history_days, 0)
    future_index = np.arange(0, len(predictions))

    if all_predictions is not None:
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 - (1 - confidence_level) / 2) * 100
        lower_bound = np.percentile(all_predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(all_predictions, upper_percentile, axis=0)
    else:
        recent_returns = df["close"].pct_change().tail(100).std()
        z_score = 1.645
        lower_bound = predictions * (1 - z_score * recent_returns)
        upper_bound = predictions * (1 + z_score * recent_returns)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(
        historical_index,
        historical_close,
        "b-",
        linewidth=2,
        label="Historical Close",
        alpha=0.8,
    )
    ax1.plot(
        future_index, predictions, "r-", linewidth=2.5, label="Forecast", alpha=0.9
    )
    ax1.fill_between(future_index, lower_bound, upper_bound,
                    color='red', alpha=0.2,
                    label=f'{int(confidence_level*100)}% Confidence Interval')
    ax1.axvline(
        x=0, color="black", linestyle="--", linewidth=2, label="Present", alpha=0.7
    )

    ax1.set_xlabel("Days (relative to present)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Close Price", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Stock Price Forecast with {int(confidence_level*100)}% Confidence Interval\n"
        f"Last {history_days} Days History + {len(predictions)} Days Forecast",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(loc="best", fontsize=11)
    ax1.grid(True, alpha=0.3)

    last_price = historical_close[-1]
    first_pred = predictions[0]
    last_pred = predictions[-1]
    total_change = ((last_pred / last_price) - 1) * 100

    stats_text = f"Last Historical: ${last_price:.2f}\n"
    stats_text += f"First Forecast: ${first_pred:.2f}\n"
    stats_text += f"Final Forecast: ${last_pred:.2f}\n"
    stats_text += f"Total Change: {total_change:+.2f}%"

    ax1.text(
        0.02,
        0.98,
        stats_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    forecast_returns = np.diff(predictions) / predictions[:-1] * 100
    ax2.bar(future_index[1:], forecast_returns, color='green', alpha=0.6,
           label='Forecasted Daily Returns')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Days (relative to present)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Daily Return (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Forecasted Daily Returns', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.savefig("forecast_with_confidence.png", dpi=150, bbox_inches="tight")
    # print("\n✓ Forecast plot saved as 'forecast_with_confidence.png'")
    st.pyplot(fig)

    # print("\n" + "=" * 70)
    # print("FORECAST SUMMARY STATISTICS")
    # print("=" * 70)
    # print(f"Historical Period: {history_days} days")
    # print(f"Forecast Period: {len(predictions)} days")
    # print(f"Confidence Level: {confidence_level*100:.0f}%")
    # print(f"\nPrice Statistics:")
    # print(f"  Last Historical Price: ${last_price:.2f}")
    # print(
    #     f"  First Forecast: ${first_pred:.2f} ({((first_pred/last_price)-1)*100:+.2f}%)"
    # )
    # print(f"  Final Forecast: ${last_pred:.2f} ({total_change:+.2f}%)")
    # print(f"  Forecast Mean: ${np.mean(predictions):.2f}")
    # print(f"  Forecast Std: ${np.std(predictions):.2f}")
    # print(f"  Forecast Min: ${np.min(predictions):.2f}")
    # print(f"  Forecast Max: ${np.max(predictions):.2f}")
    # print(f"\nConfidence Interval:")
    # print(f"  Final Lower Bound: ${lower_bound[-1]:.2f}")
    # print(f"  Final Upper Bound: ${upper_bound[-1]:.2f}")
    # print(f"  Interval Width: ${upper_bound[-1] - lower_bound[-1]:.2f}")

    # if all_predictions is not None:
    #     model_agreement = np.std(all_predictions[:, -1])
    #     print(f"\nModel Agreement:")
    #     print(f"  Std Dev of Final Predictions: ${model_agreement:.2f}")
    #     print(
    #         f"  Coefficient of Variation: {(model_agreement/np.mean(all_predictions[:, -1]))*100:.2f}%"
    #     )
