import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

from qwen_lora.utility.processing import (
    load_and_preprocess,
    load_volterra,
    post_processing,
    split_into_xy,
    tokenize_data,
)
from qwen_lora.model import QwenLoRATimeSeries


def main():
    # Import Data
    data_fp = os.path.join("data", "lotka_volterra_data.h5")
    _, _, times = load_volterra(data_fp=data_fp)
    train, val, test, _, _, test_alphas = load_and_preprocess(data_fp=data_fp)

    # Create model
    final_model = QwenLoRATimeSeries(
        learning_rate=1e-4,
        lora_rank=8,
        decimal_places=2,
        random_seed=42,
        name="final_model",
    )

    # Train (with Early Stopping)
    final_model.train(
        train_texts=train,
        val_texts=val,
        batch_size=4,
        num_steps=10000,
        track=True,
        log_interval=100,
        early_stopping=1,
        save_dir=os.path.join("data", "models", "final_model"),
    )

    # Plot the gradients
    final_model.plot_gradients(
        output_dir=os.path.join("report", "figures", "final_gradients.png")
    )

    # Plot the loss curves
    final_model.plot_training_loss(
        output_dir=os.path.join("report", "figures", "final_loss_curves.png")
    )

    # Evaluate the MSE
    # Split the test data into X and y
    X_test, y_test, len_X_test, _ = split_into_xy(test)

    # Tokenize the X to be put into the model
    X_test_tokenized = tokenize_data(data=X_test, tokenizer=final_model.tokenizer)

    # Predict using the test values on the final model
    y_pred_encoded = final_model.predict(
        tokenized_input=X_test_tokenized, max_new_predictions=25, temperature=0.2
    )

    # Post process the output sequence
    y_pred_decoded = post_processing(
        tokens=y_pred_encoded, alphas=test_alphas, dp=2, feature_range=(0, 9)
    )

    # Decode the full test set
    test_decoded = post_processing(
        tokens=test, alphas=test_alphas, dp=2, feature_range=(0, 9)
    )

    # Decode the test y values for MSE calculation
    y_test_decoded = post_processing(
        tokens=y_test, alphas=test_alphas, dp=2, feature_range=(0, 9)
    )

    # Calculate MSEs
    mses = []
    for i, test in y_test_decoded:
        prey_mse = mean_squared_error(test[0], y_pred_decoded[i][0])
        pred_mse = mean_squared_error(test[1], y_pred_decoded[i][1])

        mses.append(pred_mse)
        mses.append(prey_mse)

    plot_mse_hist(mses, os.path.join("report", "figures", "final_mses.png"))

    # Plot some examples
    _, axes = plt.subplots(ncols=4, nrows=4, figsize=(15, 10))
    axes_flat = axes.ravel()

    for i in range(len(axes_flat)):
        # Plot the actual (test) data
        axes_flat[i].plot(times, test_decoded[i][0], label="Actual Prey", color="blue")
        axes_flat[i].plot(
            times, test_decoded[i][1], label="Actual Predator", color="red"
        )

        # Plot the predicted data
        axes_flat[i].plot(
            times[len_X_test:],
            y_pred_decoded[i][0][: len(times[len_X_test:])],
            linestyle="--",
            label="Predicted Prey",
            color="blue",
        )
        axes_flat[i].plot(
            times[len_X_test:],
            y_pred_decoded[i][1][: len(times[len_X_test:])],
            linestyle="--",
            label="Predicted Predator",
            color="red",
        )

        axes_flat[i].set_title(f"Test Sample {i + 1}")
        axes_flat[i].set_xlabel("Time")
        axes_flat[i].set_ylabel("Population")
        axes_flat[i].legend()
        axes_flat[i].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join("report", "figures", "final_examples.png"))
    plt.show()


def plot_mse_hist(mses, output_dir):
    _, ax = plt.subplots(figsize=(10, 6))
    ax.hist(mses, bins=200)
    ax.axvline(x=np.mean(mses), color="red", linestyle="--")
    ax.axvline(x=np.median(mses), color="green", linestyle="--")
    ax.set_xlabel("MSE")
    ax.set_ylabel("Frequency")

    ax.legend(["Mean", "Median"])
    plt.tight_layout()
    plt.savefig(output_dir)
    plt.show()
    print(f"The mean MSE is {np.mean(mses)}")
    print(f"The median MSE is {np.median(mses)}")
