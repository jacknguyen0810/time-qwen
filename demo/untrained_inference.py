import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from accelerate import Accelerator

from qwen_lora.utility.processing import (
    load_volterra,
    load_and_preprocess,
    split_into_xy,
    tokenize_data,
    post_processing,
    calculate_mse,
    calculate_mae
)
from qwen_lora.qwen import load_qwen


# Prediction Function
def untrained_qwen_inference(
    tokenized_input,
    model,
    tokenizer,
    max_new_predictions=20,
    dp: int = 2,
    temperature: float = 0.1,
):
    """
    Use the untrained Qwen model to generate predictions for tokenized data.

    Args:
        tokenized_input: Tokenized data (list of tokenized sequences)
        model: Pretrained Qwen model
        tokenizer: Qwen tokenizer
        max_new_tokens: Maximum number of tokens to generate for each sequence
        temperature: Controls randomness in generation (lower = more deterministic)

    Returns:
        List of generated predictions (decoded text)
    """
    # Use accelerator
    accelerator = Accelerator()
    accelerator.prepare(model)

    # Ensure model is in evaluation mode
    model.eval()

    # Store all predictions
    all_predictions = []

    # Calculate how many tokens each prediction is worth:
    max_new_tokens = (2 * (2 + dp + 1)) * max_new_predictions - 1

    # Generate predictions for each input sequence
    with torch.no_grad():
        for input_ids in tokenized_input:
            # Make sure input is on the same device as model
            input_ids = input_ids.to(model.device)

            # Generate text
            outputs = model.generate(
                input_ids=input_ids.unsqueeze(0),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Get only the newly generated tokens (excluding input tokens)
            generated_tokens = outputs[0, input_ids.shape[0] :]

            # Decode the generated tokens
            prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            all_predictions.append(prediction)

    return all_predictions


# Plot the historgrams of the MSEs
def plot_mse_hist(mses):
    _, ax = plt.subplots(figsize=(10, 6))
    ax.hist(mses, bins=50)
    ax.axvline(x=mses.mean(), color="red", linestyle="--")
    ax.axvline(x=np.median(mses), color="green", linestyle="--")
    ax.set_xlabel("MSE")
    ax.set_ylabel("Frequency")

    ax.legend(["Mean", "Median"])
    plt.tight_layout()
    plt.savefig(os.path.join("report", "figures", "baseline_hist.png"))
    plt.show()
    print(f"The mean MSE is {mses.mean()}")
    print(f"The median MSE is {np.median(mses)}")


def plot_error_distributions(mses, maes, output_dir):
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot MSE distribution
    ax1.hist(mses, bins=50, color='#1f77b4', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axvline(x=np.mean(mses), color='#d62728', linestyle='--', linewidth=2, label=f'Mean: {np.mean(mses):.3f}')
    ax1.axvline(x=np.median(mses), color='#2ca02c', linestyle='--', linewidth=2, label=f'Median: {np.median(mses):.3f}')
    ax1.set_xlabel('Mean Square Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('MSE Distribution')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Plot MAE distribution
    ax2.hist(maes, bins=50, color='#ff7f0e', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axvline(x=np.mean(maes), color='#d62728', linestyle='--', linewidth=2, label=f'Mean: {np.mean(maes):.3f}')
    ax2.axvline(x=np.median(maes), color='#2ca02c', linestyle='--', linewidth=2, label=f'Median: {np.median(maes):.3f}')
    ax2.set_xlabel('Mean Absolute Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('MAE Distribution')
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir, bbox_inches='tight', dpi=300)
    plt.close()

    # Print statistics
    print("\nError Statistics:")
    print(f"MSE - Mean: {np.mean(mses):.4f}, Median: {np.median(mses):.4f}")
    print(f"MAE - Mean: {np.mean(maes):.4f}, Median: {np.median(maes):.4f}")

def main(mini=True):
    # Import and load data
    data_fp = os.path.join("data", "lotka_volterra_data.h5")
    _, _, times = load_volterra(data_fp=data_fp)
    _, val, _, _, val_alphas, _ = load_and_preprocess(
        data_fp=data_fp, return_alphas=True
    )

    # Load untrained Qwen
    model, tokenizer = load_qwen()

    # Split the data into x and y for validation
    X_val, y_val, _, _ = split_into_xy(val)

    X_val_tokenized = tokenize_data(X_val, tokenizer=tokenizer)

    if mini:
        # Perform inference on a small subset of data and plot it
        # Use the validation cases
        mini_pred_set = val[0:9]

        # Split the mini_set into x and y
        mini_pred_X, mini_val_y, len_mini_pred_X, len_mini_pred_y = split_into_xy(
            mini_pred_set
        )
        print(len_mini_pred_X)
        print(len_mini_pred_y)

        # Tokenise the mini_set
        mini_X_tokenized = tokenize_data(mini_pred_X, tokenizer=tokenizer)

        # Feed into the prediction
        mini_pred = untrained_qwen_inference(
            tokenized_input=mini_X_tokenized,
            model=model,
            tokenizer=tokenizer,
            max_new_predictions=25,
            dp=3,
            temperature=0.2,
        )

        # Postproccess the data
        mini_pred_set_decoded = post_processing(mini_pred_set, val_alphas[0:9])
        mini_val_y_decoded = post_processing(mini_val_y, val_alphas[0:9])
        mini_pred_decoded = post_processing(mini_pred, val_alphas[0:9])

        # Plotting
        _, axes = plt.subplots(ncols=3, nrows=3, figsize=(12, 8))
        axes_flat = axes.ravel()
        for i in range(len(axes_flat)):
            # Plot the x values
            # Prey
            axes_flat[i].plot(
                times, mini_pred_set_decoded[i][0], label="X (Prey)", color="red"
            )
            # Predator
            axes_flat[i].plot(
                times, mini_pred_set_decoded[i][1], label="X (Predator)", color="green"
            )

            # Plot predictions
            axes_flat[i].plot(
                times[len_mini_pred_X + 1 :],
                mini_pred_decoded[i][0][:len_mini_pred_y],
                linestyle="--",
                label="Y_pred (Prey)",
                color="red",
            )
            axes_flat[i].plot(
                times[len_mini_pred_X + 1 :],
                mini_pred_decoded[i][1][:len_mini_pred_y],
                linestyle="--",
                label="Y_pred (Predator)",
                color="green",
            )

            print(f"Number of Actual Y datapoints = {len(mini_val_y_decoded[i][0])}")
            print(f"Number of Predicted Y datapoints = {len(mini_pred_decoded[i][0])}")

        plt.tight_layout()
        plt.show()

    # Get the baseline
    # Calculate performance for all validation data-points
    y_pred = untrained_qwen_inference(
        tokenized_input=X_val_tokenized,
        model=model,
        tokenizer=tokenizer,
        max_new_predictions=25,
        dp=2,
    )

    # Postprocess the outputs:
    y_val_decoded = post_processing(y_val, val_alphas)
    y_pred_decoded = post_processing(y_pred, val_alphas)

    # Calculate both MSE and MAE
    baseline_mse = calculate_mse(y_val_decoded, y_pred_decoded)
    baseline_mae = calculate_mae(y_val_decoded, y_pred_decoded)

    # Plot error distributions
    plot_error_distributions(baseline_mse, baseline_mae, os.path.join("report", "figures"))


if __name__ == "__main__":
    # Get just the baseline
    main(mini=False)
