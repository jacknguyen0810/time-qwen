def calculate_forward_flops(seq_length: int, r: int):
    """
    Calculate the approximate number of FLOPs for a forward pass of the Qwen model.

    Primitive operation costs:
    - Addition/Subtraction/Negation = 1 FLOP
    - Multiplication/Division/Inverse = 1 FLOP
    - ReLU/Absolute Value = 1 FLOP
    - Exponent/Logarithm = 10 FLOPs
    - Sine/Cosine/Square Root = 10 FLOPs

    Args:
        seq_length: int: The input sequence length
        r: int: The LoRA rank

    Returns:
        total_flops: Estimated number of FLOPs for a forward pass
    """

    # Model Params
    hidden_size = 896
    num_layers = 24
    num_heads = 14

    q_dim = hidden_size // num_heads
    kv_dim = hidden_size // num_heads
    intermediate_size = 4864
    vocab_size = 151936

    # Calculate FLOPs for each component using formula m * p * (2 * n-1)

    # # Rotary position embeddings
    # # Each position requires sin and cos operations for each dimension
    # # For each token and each head, we need sine/cosine operations
    # # 10 FLOPs per sin/cos operation
    # rotary_flops = seq_length * q_dim * 2 * 10  # 2 for sin and cos

    # No need to calculate positional embeddings, only the cost of adding them on ()
    embedding_flops = seq_length * hidden_size

    # 1. Self-attention mechanism (per layer)
    # Query projection: Matrix multiplication of (seq_length × q_dim) with (q_dim × hidden_size) + bias (q_dim * seq_length)
    q_proj_flops = seq_length * q_dim * (2 * hidden_size - 1) + (
        seq_length + hidden_size
    )

    # LoRA on queries
    q_proj_flops += q_dim * q_dim * 2 * r

    # Key projection: Matrix multiplication of (seq_length × hidden_size) with (hidden_size × k_dim) + bias (q_dim * seq_length)
    k_proj_flops = seq_length * kv_dim * (2 * hidden_size - 1) + (
        seq_length + hidden_size
    )

    # Value projection: Matrix multiplication of (seq_length × hidden_size) with (hidden_size × v_dim) + bias (q_dim * seq_length)
    v_proj_flops = seq_length * kv_dim * (2 * hidden_size - 1) + (
        seq_length + hidden_size
    )

    # LoRA on values
    v_proj_flops += kv_dim * kv_dim * 2 * r

    # Attention score computation (QK^T): Matrix multiplication per head
    # (seq_length * head_dim) * (head_dim * seq_length) = seq_length * seq_length matrix
    attn_score_flops = seq_length * seq_length * (2 * q_dim - 1)

    # Scaling attention scores: One division per element + the square root (scaled self attention)
    scaling_flops = seq_length * seq_length * 1 + 10

    # Attention weight computation (softmax):
    softmax_flops = seq_length * (
        # Exp for each element (10 FLOPs each)
        seq_length * 10
        +
        # Sum over sequence dimension (seq_length-1 additions) for each row (remember that there are exponentials in th summation (adding then exp))
        (seq_length - 1)
        +
        # Division by sum for each element (1 FLOP each)
        seq_length * 1
    )

    # Output projection: Matrix multiplication of (q_dim * seq_length) with (seq_length * seq_length)
    output_proj_flops = q_dim * seq_length * (2 * seq_length - 1)

    # Attention Combination: (hidden_size * seq_length) with (seq_length * seq_length)
    attn_output_flops = q_dim * seq_length * (2 * seq_length - 1)

    # Total attention FLOPs per layer
    attention_flops = (
        q_proj_flops
        + k_proj_flops
        + v_proj_flops
        + attn_score_flops
        + scaling_flops
        + softmax_flops
        + attn_output_flops
        + output_proj_flops
        + embedding_flops
    )

    # 2. Feed-forward network (per layer)

    # gate_proj: Matrix multiplication of (seq_length * hidden_size) with (hidden_size * intermediate_size)
    gate_proj_flops = seq_length * intermediate_size * (2 * hidden_size - 1)

    # up_proj: Matrix multiplication of (seq_length * hidden_size) with (hidden_size * intermediate_size)
    up_proj_flops = seq_length * intermediate_size * (2 * hidden_size - 1)

    # SiLU activation: sigmoid(x) * x
    # Sigmoid involves exp (10 FLOPs), addition (1 FLOP), and division (1 FLOP)
    # Then multiplication with x (1 FLOP)
    silu_flops = seq_length * intermediate_size * (10 + 1 + 1 + 1)

    # Element-wise multiplication: 1 FLOP per element
    elementwise_mult_flops = seq_length * intermediate_size * 1

    # down_proj: Matrix multiplication of (seq_length * intermediate_size) with (intermediate_size * hidden_size)
    down_proj_flops = seq_length * hidden_size * (2 * intermediate_size - 1)

    # 3. RMSNorm operations
    # For each token and feature:
    # - Square each element (1 FLOP)
    # - Sum squared values (hidden_size-1 additions)
    # - Multiply by 1/hidden_size (1 FLOP)
    # - Square root (10 FLOPs)
    # - Divide each element by RMS (1 FLOP)
    # - Multiply by scale and add epsilon (2 FLOPs)
    rms_norm_flops_per_token = (
        hidden_size * 1  # Square each element
        + (hidden_size - 1)  # Sum squared values
        + 1  # Multiply by 1/hidden_size
        + 10  # Square root
        + hidden_size * 1  # Divide each element by RMS
        + hidden_size * 2  # Multiply by scaling factor and add epsilon
    )
    # Two RMSNorms per layer for each token
    rms_norm_flops = 2 * seq_length * rms_norm_flops_per_token

    # 4. Residual connections: 1 addition per element
    residual_flops = 2 * seq_length * hidden_size * 1

    # Total FLOPs per layer
    flops_per_layer = (
        attention_flops
        + gate_proj_flops
        + up_proj_flops
        + silu_flops
        + elementwise_mult_flops
        + down_proj_flops
        + rms_norm_flops
        + residual_flops
    )

    # 5. Final RMSNorm
    final_norm_flops = seq_length * rms_norm_flops_per_token

    # 6. Output layer (language modeling head)
    # Matrix multiplication of (seq_length × hidden_size) with (hidden_size * vocab_size)
    lm_head_flops = seq_length * vocab_size * (2 * hidden_size - 1)

    # Total FLOPs
    total_flops = (flops_per_layer * num_layers) + final_norm_flops + lm_head_flops

    return total_flops
