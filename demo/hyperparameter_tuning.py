import os
import json

import pandas as pd
from pandas.plotting import parallel_coordinates
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from qwen_lora.utility.processing import load_and_preprocess
from qwen_lora.model import QwenLoRATimeSeries


def perform_tuning():
    # Load data
    # Get data
    data_fp = os.path.join("data", "lotka_volterra_data.h5")
    train, val, _, _, _, _ = load_and_preprocess(data_fp=data_fp, return_alphas=True)

    # Get the hyperparameter set
    tuning_train = train[200:250]
    tuning_val = val[75:85]

    hyperparameter_results, tuning_flops = hyperparameter_tuning(
        train_texts=tuning_train,
        val_texts=tuning_val,
        lora_ranks=[2, 4, 8],
        learning_rates=[1e-4, 5e-5, 1e-5],
        num_steps=600,
        batch_size=4,
        save_dir=os.path.join("data", "models", "tuning_best"),
    )

    print(f"Total Hyperparameter FLOPS: {tuning_flops:.4e}")
    plot_optimization_results(
        hyperparameter_results,
        os.path.join("report", "figures", "hyperparameter_results.png"),
    )


def hyperparameter_tuning(
    train_texts,
    val_texts,
    lora_ranks=None,
    learning_rates=None,
    num_steps=600,
    batch_size=4,
    save_dir="lora_tuning_results",
):
    """
    Performs hyperparameter tuning on the QwenLoRATimeSeries model

    Args:
        train_texts (List[str]): List of strings to trian the model, each entry is a 2 pair system
        val_texts (List[str]): List of strings to validate the model, each entry is a 2 pair system
        lora_ranks (List[int], optional): LoRA ranks to test. Defaults to None.
        learning_rates (List[int], optional): Learning rates to test. Defaults to None.
        num_steps (int, optional): Number of training steps. Defaults to 600.
        batch_size (int, optional): Training batch size. Defaults to 4.
        save_dir (str, optional): Filepath to save the best model. Defaults to "lora_tuning_results".

    Returns:
        dict: Results
        int: FLOPS used
    """
    lora_ranks = lora_ranks or [2, 4, 8]
    learning_rates = learning_rates or [1e-4, 5e-5, 1e-5]

    os.makedirs(save_dir, exist_ok=True)
    tuning_results = {}
    best_val_loss = float("inf")  # Initialize with a very large value
    best_model_path = None
    total_flops = 0

    for lora_rank in lora_ranks:
        for learning_rate in learning_rates:
            model = QwenLoRATimeSeries(
                lora_rank=lora_rank,
                learning_rate=learning_rate,
            )
            flops = model.train(
                max_context_length=512,
                train_texts=train_texts,
                val_texts=val_texts,
                num_steps=num_steps,
                batch_size=batch_size,
                save_dir=save_dir,
                track=False,
            )

            total_flops += flops

            tuning_results[f"lora_rank_{lora_rank}_lr_{learning_rate}"] = {
                "lora_rank": lora_rank,
                "learning_rate": learning_rate,
                "val_loss": model.val_loss,
                "flops": model.flops,
            }

            # Check if this model is the best so far
            if model.val_loss < best_val_loss:
                best_val_loss = model.val_loss
                best_model_path = os.path.join(
                    save_dir, model.name
                )  # Path to the best model

                # Remove previous best model (if any) to save space
                if (
                    best_model_path is not None
                    and os.path.exists(best_model_path)
                    and best_model_path != os.path.join(save_dir, model.name)
                ):
                    import shutil

                    shutil.rmtree(
                        best_model_path
                    )  # Remove the directory and its contents

    # Save tuning results
    with open(
        os.path.join(save_dir, "tuning_results.json"),
        "w",
        encoding="utf-8",
        encoding="utf-8",
    ) as f:
        json.dump(tuning_results, f, indent=4)

    # Save information about the best model
    with open(
        os.path.join(save_dir, "best_model_info.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {"best_model_path": best_model_path, "best_val_loss": best_val_loss},
            f,
            indent=4,
        )

    return tuning_results, total_flops


def plot_optimization_results(results, output_dir):
    # Convert results to a DataFrame
    trials_data = pd.DataFrame.from_dict(results, orient="index")
    trials_data = trials_data.reset_index(names="trial")  # Add a trial column

    # Extract trial data (Modified to work with the existing results format)
    trials_data = pd.DataFrame(
        [
            {
                "trial": trial_name,  # Use trial name as index
                "learning_rate": trial_data["learning_rate"],
                "lora_rank": trial_data["lora_rank"],
                "val_loss": trial_data["val_loss"],
            }
            for trial_name, trial_data in results.items()
        ]
    )

    # Create figure with subplots
    _, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot validation loss over trials
    sns.scatterplot(
        data=trials_data, x=[i for i in range(9)], y="val_loss", ax=axes[0, 0]
    )
    axes[0, 0].set_title("Validation Loss per Trial")
    # axes[0,0].set_yscale("log")

    # Plot learning rate vs validation loss
    sns.boxplot(data=trials_data, x="learning_rate", y="val_loss", ax=axes[0, 1])
    axes[0, 1].set_title("Learning Rate vs Validation Loss")
    # axes[0,1].set_xlim([5e-6, 5e-4])
    # axes[0,1].set_xscale("log")
    axes[0, 1].set_yscale("log")

    # Plot LoRA rank vs validation loss
    sns.boxplot(data=trials_data, x="lora_rank", y="val_loss", ax=axes[1, 0])
    axes[1, 0].set_title("LoRA Rank vs Validation Loss")
    axes[1, 0].set_yscale("log")

    # Plot parameter parallel coordinates
    # Normalize data for parallel coordinates
    parallel_data = trials_data.copy()
    parallel_data["learning_rate"] = np.log10(parallel_data["learning_rate"])
    for col in ["learning_rate", "lora_rank", "val_loss"]:
        parallel_data[col] = (parallel_data[col] - parallel_data[col].min()) / (
            parallel_data[col].max() - parallel_data[col].min()
        )

    # Specify class_column to color lines based on a variable
    # Change this to match the structure of your trials_data:
    parallel_coordinates(
        parallel_data,
        class_column="lora_rank",  # Assign lora_rank as the grouping variable
        cols=[
            "learning_rate",
            "lora_rank",
            "val_loss",
        ],  # Specifying the features to plot
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Parameter Parallel Coordinates")

    plt.tight_layout()
    plt.savefig(output_dir)
    plt.show()


if __name__ == "__main__":
    perform_tuning()
