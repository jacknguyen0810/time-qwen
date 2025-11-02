# Import libraries
import json
import datetime

import pandas as pd
import matplotlib.pyplot as plt


import torch
from tqdm import tqdm
from accelerate import Accelerator

from qwen_lora.lora_linear import LoRALinear
from qwen_lora.qwen import load_qwen
from qwen_lora.utility.flops import calculate_forward_flops


class QwenLoRATimeSeries:
    """
    Performs time series prediction using Qwen adapte with a LoRA implementation. 
    """
    def __init__(
        self,
        learning_rate: float = 1e-5,
        lora_rank: int = 4,
        decimal_places: int = 2,
        random_seed: int = 42,
        name: str = None,
    ):
        """_summary_

        Args:
            learning_rate (float, optional): Learning rate for optimiser. Defaults to 1e-5.
            lora_rank (int, optional): Number of ranks for LoRA. Defaults to 4.
            decimal_places (int, optional): Number of decimal places of the time series itself. Defaults to 2.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
            name (str, optional): Model name. Defaults to None.
        """

        # Each instance of Qwen-LoRA will be a single model
        self.seed = random_seed
        self.dp = decimal_places
        self.training_stats = None
        self.flops = 0
        self.val_loss = None
        self.rank = lora_rank

        if name is None:
            self.name = (
                f"qwen_lora_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
            )
        else:
            self.name = name

        # Import Qwen and set model and tokenizer
        self.model, self.tokenizer = load_qwen()

        # Apply the LoRA Architecture
        for layer in self.model.model.layers:
            layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=lora_rank)
            layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=lora_rank)

        # Set Adam as optimiser
        self.optimizer = torch.optim.Adam(
            (p for p in self.model.parameters() if p.requires_grad), lr=learning_rate
        )

    # ---------- TRAINING FUNCTIONS -----------
    def train(
        self,
        train_texts,
        val_texts,
        max_context_length=512,
        batch_size=1,
        num_steps=10000,
        stride=None,
        track=False,
        log_interval=10,
        early_stopping=None,
        save_dir=None,
    ):
        """Training function for the model. 

        Args:
            train_texts (list[str]): List of strings for training
            val_texts (list[str]): List of strings for validation
            max_context_length (int, optional): Context window size for learning. Defaults to 512.
            batch_size (int, optional): Batch size for training. Defaults to 1.
            num_steps (int, optional): Number of training steps. Defaults to 10000.
            stride (int, optional): Stride length for processing the training and validation data. Defaults to None.
            track (bool, optional): If on, the model is tracked and validated. Defaults to False.
            log_interval (int, optional): Logging interval for tracking and validation. Defaults to 10.
            early_stopping (int, optional): If None, no early stepping, else the value is the early stopping patience. Defaults to None.
            save_dir (str, optional): String pointing to the directory to save the information about training. Defaults to None.

        Returns:
            _type_: _description_
        """
        # Set stride length
        if stride is None:
            stride = max_context_length // 2

        # Process the training and validation data
        train_input_ids = self.process_sequences(
            texts=train_texts,
            tokenizer=self.tokenizer,
            max_context_length=max_context_length,
            stride=stride,
        )

        val_input_ids = self.process_sequences(
            texts=val_texts,
            tokenizer=self.tokenizer,
            max_context_length=max_context_length,
            stride=stride,
        )

        # DataLoaders
        train_dataset = torch.utils.data.TensorDataset(train_input_ids)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        val_dataset = torch.utils.data.TensorDataset(val_input_ids)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

        # Use accelerate
        accelerator = Accelerator()
        self.model, self.optimizer, train_loader = accelerator.prepare(
            self.model, self.optimizer, train_loader
        )

        # Set up tracking tools
        if track:
            # Dictionary to hold values for tracking
            self.training_stats = {
                "train_loss": [],
                "val_loss": [],
                "train_steps": [],
                "val_steps": [],
                "time": [],
                "best_val_loss": float("inf"),
                "best_step": 0,
                "lora_gradients": {
                    "layer_0": {  # Track by layer
                        "q_proj": {"A": [], "B": []},
                        "v_proj": {"A": [], "B": []},
                    }
                    # Will be expanded for each layer
                },
            }
            running_loss = 0

        # Early Stopping setup
        if early_stopping is not None:
            best_val_loss = float("inf")
            patience_counter = 0
            best_step = 0

        # Training Loop
        # Put model into training mode
        self.model.train()
        steps = 0

        early_stop = False

        while steps < num_steps and not early_stop:
            progress_bar = tqdm(train_loader, desc=f"Steps {steps}")
            for batch_idx, (batch,) in enumerate(progress_bar):
                # Send batch to device being used
                batch = batch.to(self.model.device)

                # Forward Pass
                # Set optimiser to zero gradient
                self.optimizer.zero_grad()
                # Get outputs
                outputs = self.model(batch, labels=batch)
                # Get loss
                loss = outputs.loss

                # Use Accelerator for backprop
                accelerator.backward(loss)

                # Track gradients before weights and biases change
                if track and steps % log_interval == 0:
                    for layer_idx, layer in enumerate(self.model.model.layers):
                        # Initialize layer tracking if not exists
                        if (
                            f"layer_{layer_idx}"
                            not in self.training_stats["lora_gradients"]
                        ):
                            self.training_stats["lora_gradients"][
                                f"layer_{layer_idx}"
                            ] = {
                                "q_proj": {"A": [], "B": []},
                                "v_proj": {"A": [], "B": []},
                            }

                        # Track Q projection gradients
                        if (
                            hasattr(layer.self_attn.q_proj, "A")
                            and layer.self_attn.q_proj.A.grad is not None
                        ):
                            self.training_stats["lora_gradients"][f"layer_{layer_idx}"][
                                "q_proj"
                            ]["A"].append(layer.self_attn.q_proj.A.grad.norm().item())
                            self.training_stats["lora_gradients"][f"layer_{layer_idx}"][
                                "q_proj"
                            ]["B"].append(layer.self_attn.q_proj.B.grad.norm().item())

                        # Track V projection gradients
                        if (
                            hasattr(layer.self_attn.v_proj, "A")
                            and layer.self_attn.v_proj.A.grad is not None
                        ):
                            self.training_stats["lora_gradients"][f"layer_{layer_idx}"][
                                "v_proj"
                            ]["A"].append(layer.self_attn.v_proj.A.grad.norm().item())
                            self.training_stats["lora_gradients"][f"layer_{layer_idx}"][
                                "v_proj"
                            ]["B"].append(layer.self_attn.v_proj.B.grad.norm().item())

                # Optimize loss to change variables (update weghts and biases)
                self.optimizer.step()

                # Validation and Early Stopping Check
                if steps % log_interval == 0:
                    val_loss = self.evaluate_model(self.model, val_loader)

                    if track:
                        self.training_stats["val_loss"].append(val_loss)
                        self.training_stats["val_steps"].append(steps)

                    if early_stopping is not None:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                            best_step = steps

                            if track:
                                self.training_stats["best_val_loss"] = best_val_loss
                                self.training_stats["best_step"] = best_step
                        else:
                            patience_counter += 1
                            if patience_counter >= early_stopping:
                                print(
                                    f"\nEarly stopping triggered. Best val_loss: {best_val_loss:.6f}"
                                )
                                early_stop = True
                                break

                if track:
                    # Update the running loss
                    running_loss += loss.item()

                    # Update variables
                    self.training_stats["train_loss"].append(
                        running_loss / (batch_idx + 1)
                    )
                    self.training_stats["train_steps"].append(steps)

                    # Calculate validation loss
                    if steps % log_interval == 0:
                        self.training_stats["val_loss"].append(val_loss)
                        self.training_stats["val_steps"].append(steps)

                    # Get the best step (validation loss)
                    if val_loss < self.training_stats["best_val_loss"]:
                        self.training_stats["best_val_loss"] = val_loss
                        self.training_stats["best_step"] = steps

                    # Put the model back to train model
                    self.model.train()

                # Next training step:
                steps += 1

                # Update progress bar
                progress_bar.set_postfix(loss=loss.item())
                if steps >= num_steps:
                    break

            if track:
                # Reset the running loss
                running_loss = 0.0

        # Final evaluation
        val_loss = self.evaluate_model(self.model, val_loader)
        if track:
            self.training_stats["val_loss"].append(val_loss)
            self.training_stats["val_steps"].append(steps)

        # Set model back to eval mode:
        self.model.eval()

        if save_dir is not None:
            if track:
                self.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    tokenizer=self.tokenizer,
                    save_dir=save_dir,
                    training_stats=self.training_stats,
                )
            else:
                self.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    tokenizer=self.tokenizer,
                    save_dir=save_dir,
                    training_stats=None,
                )

        self.val_loss = val_loss

        # Calculate the approximate number of flops
        # Training FLOPS
        flops = (
            calculate_forward_flops(max_context_length, self.rank)
            * 3
            * batch_size
            * steps
        )
        print(f"{flops} FLOPS used in training")

        # Validation FLOPS
        if track:
            flops += (
                calculate_forward_flops(max_context_length, self.rank)
                * batch_size
                * (steps // log_interval)
            )
        else:
            flops += calculate_forward_flops(max_context_length, self.rank)
        print(f"{flops} FLOPS used in training and validation")

        self.flops += flops

        return flops

    # Modified tokenization with chunking
    @staticmethod
    def process_sequences(tokenizer, texts, max_context_length=512, stride=256):
        all_input_ids = []
        for text in texts:
            # Apply Qwen's tokenization scheme to the text:
            encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            seq_ids = encoding.input_ids[0]

            # Create sliding windows to further divide the data into chunks:
            for i in range(0, len(seq_ids), stride):
                chunk = seq_ids[i : i + max_context_length]
                if len(chunk) < max_context_length:
                    chunk = torch.cat(
                        [
                            chunk,
                            torch.full(
                                (max_context_length - len(chunk),),
                                tokenizer.pad_token_id,
                            ),
                        ]
                    )
                all_input_ids.append(chunk)
        return torch.stack(all_input_ids)

    # Validation Loss Evaluator
    @staticmethod
    def evaluate_model(model, val_loader):
        model.eval()
        total_loss = 0
        total_batches = 0

        with torch.no_grad():
            for (batch,) in tqdm(val_loader, desc="Evaluating"):
                batch = batch.to(model.device)
                outputs = model(batch, labels=batch)
                loss = outputs.loss

                total_loss += loss.item()
                total_batches += 1

        return total_loss / total_batches

    # Save model
    def save_checkpoint(
        self, model, optimizer, tokenizer, save_dir, training_stats=None
    ):
        model.save_pretrained(save_dir, max_shard_size="500MB", save_peft_format=True)
        tokenizer.save_pretrained(save_dir)

        # Save optimizer state
        torch.save(optimizer.state_dict(), f"{save_dir}.pt")

        if training_stats is not None:
            with open(f"{save_dir}/{self.name}.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "train_loss": training_stats["train_loss"],
                        "val_loss": training_stats["val_loss"],
                        "train_steps": training_stats["train_steps"],
                        "val_steps": training_stats["val_steps"],
                        "best_val_loss": training_stats["best_val_loss"],
                        "best_step": training_stats["best_step"],
                    },
                    f,
                )

    # ---------- TRAINING PLOTTING FUNCTIONS ----------------
    # Plot losses
    def plot_training_loss(self, output_dir, ema=50):
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.figsize': (12, 5),
            'figure.dpi': 300
        })

        # Import data into a Pandas dataframe
        train_stats_df = pd.DataFrame({"train_loss": self.training_stats["train_loss"]})
        train_stats_df["train_loss_ema"] = train_stats_df["train_loss"].rolling(window=ema, min_periods=1).mean()
        
        fig, ax = plt.subplots(1, 2)

        # Plot training loss
        ax[0].plot(self.training_stats["train_steps"], 
                  self.training_stats["train_loss"], 
                  color='#1f77b4', 
                  alpha=0.2, 
                  label='Raw Loss')
        ax[0].plot(self.training_stats["train_steps"], 
                  train_stats_df["train_loss_ema"], 
                  color='#1f77b4', 
                  linewidth=2, 
                  label='Moving Average')
        ax[0].set_ylabel('Training Loss')
        ax[0].set_xlabel('Training Steps')
        ax[0].grid(True, linestyle='--', alpha=0.7)
        ax[0].legend(frameon=True, fancybox=True, shadow=True)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)

        # Plot validation loss
        ax[1].plot(self.training_stats["val_steps"], 
                  self.training_stats["val_loss"], 
                  color='#ff7f0e', 
                  linewidth=2, 
                  label='Validation Loss')
        min_val_loss = min(self.training_stats["val_loss"])
        ax[1].axhline(min_val_loss, 
                     linestyle='--', 
                     color='#2ca02c', 
                     label=f'Best Loss: {min_val_loss:.4f}')
        ax[1].set_ylabel('Validation Loss')
        ax[1].set_xlabel('Training Steps')
        ax[1].grid(True, linestyle='--', alpha=0.7)
        ax[1].legend(frameon=True, fancybox=True, shadow=True)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{output_dir}_loss.png", bbox_inches='tight', dpi=300)
        plt.close()

    # Plot the gradients
    def plot_gradients(self, output_dir):
        num_layers = len(self.training_stats["lora_gradients"])

        # Create figure with subplots for each layer
        _, axes = plt.subplots(num_layers, 2, figsize=(15, 5 * num_layers))
        if num_layers == 1:
            axes = axes.reshape(1, -1)

        for layer_idx in range(num_layers):
            layer_key = f"layer_{layer_idx}"
            layer_data = self.training_stats["lora_gradients"][layer_key]

            # Determine the minimum length to avoid index errors
            min_length = min(len(self.training_stats["val_steps"]), len(layer_data["q_proj"]["A"]))

            # Plot Q projections
            axes[layer_idx, 0].plot(
                self.training_stats["val_steps"][:min_length],
                layer_data["q_proj"]["A"][:min_length],
                label="A matrix",
            )
            axes[layer_idx, 0].plot(
                self.training_stats["val_steps"][:min_length],
                layer_data["q_proj"]["B"][:min_length],
                label="B matrix",
            )
            axes[layer_idx, 0].set_title(f"Layer {layer_idx} - Q Projection Gradients")
            axes[layer_idx, 0].set_xlabel("Steps")
            axes[layer_idx, 0].set_ylabel("Gradient Norm")
            axes[layer_idx, 0].legend()

            # Plot V projections
            axes[layer_idx, 1].plot(
                self.training_stats["val_steps"][:min_length],
                layer_data["v_proj"]["A"][:min_length],
                label="A matrix",
            )
            axes[layer_idx, 1].plot(
                self.training_stats["val_steps"][:min_length],
                layer_data["v_proj"]["B"][:min_length],
                label="B matrix",
            )
            axes[layer_idx, 1].set_title(f"Layer {layer_idx} - V Projection Gradients")
            axes[layer_idx, 1].set_xlabel("Steps")
            axes[layer_idx, 1].set_ylabel("Gradient Norm")
            axes[layer_idx, 1].legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}.png")
        plt.show()

    # ----------- PREDICTION --------------
    # Prediction Function
    def predict(
        self, tokenized_input, max_new_predictions=20, temperature=0.1, batch_size=10
    ):
        """Prediction function. 

        Args:
            tokenized_input (list[list[str]]): List of strings that represent the input to the model
            max_new_predictions (int, optional): Maximum number of predicted time steps. Defaults to 20.
            temperature (float, optional): Controls creativity of predictions. Defaults to 0.1.
            batch_size (int, optional): Batch size for prediction. Defaults to 10.

        Returns:
            _type_: _description_
        """
        # Define valid tokens (numbers, decimal point, comma, semicolon)
        valid_tokens = set("0123456789.,;")

        # Get the model's vocabulary size
        vocab_size = self.model.config.vocab_size

        # Create a mask for valid tokens
        valid_token_mask = torch.zeros(vocab_size, dtype=torch.bool)

        # Mark tokens as valid if they decode to valid characters
        for token_id in range(vocab_size):
            decoded = self.tokenizer.decode([token_id])
            if any(char in valid_tokens for char in decoded):
                valid_token_mask[token_id] = True

        # Use accelerator
        accelerator = Accelerator()
        self.model = accelerator.prepare(self.model)

        # Ensure model is in evaluation mode
        self.model.eval()

        all_predictions = []
        max_new_tokens = (2 * (2 + self.dp + 1)) * max_new_predictions - 1

        with torch.no_grad():
            # Process in smaller batches (fix previous memory error)
            for i in range(0, len(tokenized_input), batch_size):
                batch_input_ids = tokenized_input[i : i + batch_size]
                input_ids = torch.stack(batch_input_ids).to(self.model.device)

                # Generate predictions
                for _ in range(max_new_tokens):
                    outputs = self.model(input_ids)
                    logits = outputs.logits[:, -1, :]

                    # Set logits for invalid tokens to -inf
                    logits[:, ~valid_token_mask] = float("-inf")

                    # Sample the next token
                    probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    # Append the predicted token to the input_ids
                    input_ids = torch.cat([input_ids, next_token], dim=-1)

                # Decode and store predictions
                for j in range(input_ids.shape[0]):
                    original_length = input_ids.shape[1] - max_new_tokens
                    generated_tokens = input_ids[j, original_length:]
                    prediction = self.tokenizer.decode(
                        generated_tokens, skip_special_tokens=True
                    )
                    all_predictions.append(prediction)

        return all_predictions
