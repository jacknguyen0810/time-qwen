import os

from qwen_lora.utility.processing import load_and_preprocess
from qwen_lora.model import QwenLoRATimeSeries


def mini_model_training():
    # Get data
    data_fp = os.path.join("data", "lotka_volterra_data.h5")
    train, val, _, _, _, _ = load_and_preprocess(data_fp=data_fp, return_alphas=True)

    train_mini = train[0:5]
    val_mini = val[0:2]

    # Initialize the QwenLoRA model with default learning rate and loRA rank
    qwen_lora_mini = QwenLoRATimeSeries(learning_rate=1e-4, lora_rank=4)
    training_flops = qwen_lora_mini.train(
        train_texts=train_mini,
        val_texts=val_mini,
        max_context_length=512,
        batch_size=4,
        num_steps=600,
        log_interval=50,
        track=True,
    )

    print(f"Training FLOPS: {training_flops:.4e}")

    # Plot the loss curves
    qwen_lora_mini.plot_training_loss(
        output_dir=os.path.join("report", "figures", "mini_loss_curves.jpeg"), ema=20
    )

    # Plot the gradients of V and K projections of each of the 24 transformer layers
    qwen_lora_mini.plot_gradients(
        output_dir="/content/drive/MyDrive/code/m2/mini_gradients"
    )

    return training_flops


if __name__ == "__main__":
    mini_train_flops = mini_model_training()
