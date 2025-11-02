import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from qwen_lora.utility.processing import load_volterra


def main():
    # Import data
    data_fp = os.path.join("data", "lotka_volterra_data.h5")
    pred, prey, times = load_volterra(data_fp=data_fp)

    # Checking data is complete:
    # For every feature, there should be 100 values, so check if these are complete:
    print(f"The number of time steps in times: {len(times)}")
    incomplete_features_prey = []
    incomplete_features_predator = []
    # Checking if each prey and predator has the full 100 values
    for i in range(1000):
        if len(prey[i, :]) != 100:
            incomplete_features_prey.append(i)
        if len(pred[i, :]) != 100:
            incomplete_features_predator.append(i)
    print(f"There are {len(incomplete_features_predator)} incomplete predator series.")
    print(f"There are {len(incomplete_features_prey)} incomplete prey series.")

    # Look at the format of the data
    # Take 10 samples from the prey and pred set and see what format it is in:
    print(f"\nSample subset of a prey sequence {prey[345, 30:35]}")
    print(f"Sample subset of a pred sequence {pred[345, 30:35]}")
    print(f"Sample subset of time steps data {times[:5]}")

    # Plot the predators and preys against time series?
    _, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 6))
    axes_flat = axes.ravel()
    for i in range(4):
        # Each row is a system
        axes_flat[i].plot(times, prey[i, :], label="prey")
        axes_flat[i].plot(times, pred[i, :], label="predator")
        axes_flat[i].legend()
        axes_flat[i].grid(True)
        axes_flat[i].set_title(f"Predator-Prey System {i}")
        axes_flat[i].set_xlabel("Time")
        axes_flat[i].set_ylabel("Population")
    plt.tight_layout()
    plt.show()

    # Plot t-SNE of data to see if there are any clear groups of data
    combined_data = np.zeros((prey.shape[0], prey.shape[1] * 2))
    combined_data[:, : prey.shape[1]] = prey
    combined_data[:, prey.shape[1] :] = pred

    tsne = TSNE(n_components=2, perplexity=50, random_state=42, n_iter=500, verbose=1)
    tnse_results = tsne.fit_transform(combined_data)

    _, ax = plt.subplots(figsize=(10, 6))
    _ = ax.scatter(tnse_results[:, 0], tnse_results[:, 1])
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_title("t-SNE Visualization of the Trajectories")

    plt.show()


if __name__ == "__main__":
    main()
