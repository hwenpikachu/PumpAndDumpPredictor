import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_and_plot(pred_csv="ENA_5min_pump_forecast_from_PEPE_model.csv"):
    df = pd.read_csv(pred_csv)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    label_col = "pump_forecast_label"

    # Helper to pick a model that actually has predictions
    def pick_model(df):
        models = [
            ("rf", "rf_pred", "rf_proba"),
            ("lr", "lr_pred", "lr_proba"),
        ]
        for name, pred_col, proba_col in models:
            if pred_col in df.columns and proba_col in df.columns:
                non_nan = df[pred_col].notna().sum()
                print(f"Model {name}: {non_nan} non-NaN predictions.")
                if non_nan > 0:
                    return name, pred_col, proba_col
        return None, None, None

    model_name, pred_col, proba_col = pick_model(df)
    if model_name is None:
        print("No model predictions found (all rf_pred/lr_pred are NaN).")
        print("This usually means the detector had no positive labels for training.")
        return

    # Test set := rows where we actually have predictions
    test_mask = df[pred_col].notna()
    test_df = df.loc[test_mask].copy()

    if test_df.empty:
        print("Test set appears empty (no predictions).")
        return

    y_true = test_df[label_col].astype(int).values
    y_pred = test_df[pred_col].astype(int).values
    y_proba = test_df[proba_col].astype(float).values

    print(f"\nUsing model: {model_name.upper()}")
    print(f"Test size: {len(test_df)} rows")
    print(f"Positives in test: {int(y_true.sum())}, negatives: {len(y_true) - int(y_true.sum())}\n")

    # === Classification report + confusion matrix ===
    print("=== Classification report (test set) ===")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)

    # --- Error margin = |y_true - predicted_probability| ---
    error_margin = np.abs(y_true - y_proba)
    test_df["proba"] = y_proba
    test_df["error_margin"] = error_margin

    # Plot 1: Confusion matrix
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm)
    ax.set_title(f"Confusion Matrix ({model_name.upper()}, Test Set)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png", dpi=200)
    plt.close()

    # Plot 2: Error margin histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(error_margin, bins=30)
    ax.set_xlabel("|y_true - predicted_probability|")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Margin Distribution ({model_name.upper()}, Test Set)")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"{model_name}_error_margin_hist.png", dpi=200)
    plt.close()

    # Plot 3: Time series of predicted probability vs true label
    if "date" in test_df.columns:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(test_df["date"], test_df["proba"], label="Predicted pump probability")

        # --- RED arrows marking true pump events ---
        ax.scatter(
            test_df.loc[test_df[label_col] == 1, "date"],
            test_df.loc[test_df[label_col] == 1, "proba"],
            marker="^", s=28, color="red", label="True pumps (actual)", zorder=5
        )

        ax.set_xlabel("Time (test region)")
        ax.set_ylabel("Probability / label")
        ax.set_title(f"{model_name.upper()} Pump Probability vs True Labels (Test Set)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{model_name}_proba_vs_true.png", dpi=200)
        plt.close()

    print("\nSaved plots:")
    print(f"- {model_name}_confusion_matrix.png")
    print(f"- {model_name}_error_margin_hist.png")
    print(f"- {model_name}_proba_vs_true.png")


if __name__ == "__main__":
    evaluate_and_plot()
