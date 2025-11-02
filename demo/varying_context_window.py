import os

import matplotlib.pyplot as plt
import numpy as np

from qwen_lora.utility.processing import load_and_preprocess
from qwen_lora.model import QwenLoRATimeSeries


def main():
    # Get data
    (
        train,
        val,
        _,
        _,
        _,
        _,
    ) = load_and_preprocess(data_fp=os.path.join("data", "lotka_volterra_data.h5"))
    ct_train = train[200:250]
    ct_val = val[5:85]

    # Perform experiment
    ct_results, ct_flops = vary_context_window(
        window_sizes=[128, 256, 768], train_texts=ct_train, val_texts=ct_val
    )

    # Plot results
    plot_context_window(
        ct_results,
        ct_flops,
        [128, 256, 768],
        os.path.join("report", "figures", "ct_window.png"),
    )

    print(f"Context Window Experiment FLOPS: {ct_flops}")


# Investigate the effect of the context window
def vary_context_window(window_sizes, train_texts, val_texts):
    results = {}
    for window in window_sizes:
        results[window] = []

    ct_flops = []

    for window_size in window_sizes:
        # Initialise model with best parameters
        model = QwenLoRATimeSeries(
            learning_rate=1e-4, lora_rank=8, name=f"context_window_{window_size}"
        )

        # Train with varying context window sizes
        flops = model.train(
            train_texts=train_texts,
            val_texts=val_texts,
            max_context_length=window_size,
            batch_size=4,
            num_steps=300,
            track=True,
            log_interval=30,
        )

        ct_flops.append(flops)
        results[window_size].extend(model.training_stats["val_loss"])
        results["time_steps"] = model.training_stats["val_steps"]

    return results, ct_flops


# Plot the effect of changing context window
def plot_context_window(context_window_results, ct_flops, window_sizes, output_dir):

    _, ax = plt.subplots(ncols=2, figsize=(12, 8))

    for window_size in window_sizes:
        results = context_window_results[window_size]
        ax[0].plot(
            context_window_results["time_steps"],
            results,
            label=f"Context Window Size = {window_size}",
        )

    ax[0].set_xlabel("Trianing Steps")
    ax[0].set_ylabel("Validation Loss")
    ax[0].set_title("Effect of Context Window Size on Validation Loss")
    ax[0].grid(True)
    ax[0].legend()

    width = 0.8
    x_pos = np.arange(len(window_sizes))

    ax[1].bar(x_pos, ct_flops, width=width)
    ax[1].set_xlabel("Context Window Size")
    ax[1].set_ylabel("FLOPS")
    ax[1].set_title("Number of FLOPS while varying context window size")
    ax[1].set_xticks(x_pos, window_sizes)

    plt.tight_layout()
    plt.savefig(output_dir)
    plt.show()


if __name__ == "__main__":
    main()
