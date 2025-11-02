import h5py
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# Load Data
def load_volterra(data_fp: str):
    """Loads the lotka_volterra dataset

    Args:
        data_fp (str): Filepath to the .h5 file

    Returns:
        np.array: Array of prey systems
        np.array: Array of predator systems
        np.array: Array of time points
    """
    with h5py.File(data_fp, "r") as f:
        # Accessing dataset
        trajectories = f["trajectories"][:]
        time_points = f["time"][:]

    # Get all time points
    prey = trajectories[:, :, 0]
    pred = trajectories[:, :, 1]
    times = time_points[:]
    return prey, pred, times


def load_and_preprocess(
    data_fp: str,
    dp: int = 2,
    random_seed: int = 42,
    alpha_range: tuple = (0, 9),
    train_val_split: float = 0.8,
    return_alphas: bool = True,
):
    """Both loads the preprocesses the Lotka Volterra dataset in a format that can be tokensied by the built-in Qwen tokeniser

    Args:
        data_fp (str): Filepath to the dataset
        dp (int, optional): Number of decimal places to round to. Defaults to 2.
        random_seed (int, optional): Deterministic seed. Defaults to 42.
        alpha_range (tuple, optional): Range to normalise values by. Defaults to (0, 9).
        train_val_split (float, optional): Split amount of overall data for training. Defaults to 0.8.
        return_alphas (bool, optional): Boolean to return the alpha scaling values. Defaults to True.
    """
    # Load the dataset
    prey, pred, _ = load_volterra(data_fp)

    # Scaling: The value of prey and predator should be scaled between 0 and 1
    systems = []
    alpha = []

    for i in range(len(prey)):
        system_text = ""
        # Combine prey and pred to make a single time series system
        single_system = np.column_stack((prey[i, :], pred[i, :]))

        # Min/Max Scaling
        scaler = MinMaxScaler(feature_range=alpha_range)
        single_system = scaler.fit_transform(single_system)
        # Return the scaling alphas
        # single_alpha = scaler.scale_
        alpha.append(scaler.scale_)

        # Go through the samples and transform each time series into a string ("prey,pred;prey,pred;...")
        for t in single_system:
            prey_val = f"{t[0]:.{dp}f}"
            pred_val = f"{t[1]:.{dp}f}"
            system_text += f"{prey_val},{pred_val};"

        systems.append(system_text[:-1])

    # Shuffle dataset
    np.random.seed(seed=random_seed)
    p = np.random.permutation(len(alpha))
    systems = np.array(systems)[p]
    alpha = np.array(alpha)[p]

    # Split the dataset into a training and validation set
    train_val_systems = np.split(systems, [int(train_val_split * len(systems))])
    val_test_systems = np.split(
        train_val_systems[1], [int(0.5 * len(train_val_systems[1]))]
    )

    if return_alphas:
        return (
            train_val_systems[0],
            val_test_systems[0],
            val_test_systems[1],
            alpha[: len(train_val_systems[0])],
            alpha[
                len(train_val_systems[0]) : len(train_val_systems[0])
                + len(val_test_systems[0])
            ],
            alpha[len(val_test_systems[0]) + len(train_val_systems[0]) :],
        )
    else:
        return train_val_systems[0], val_test_systems[0], val_test_systems[1]


# Create a decoding function
def post_processing(tokens: list, alphas: list, dp: int = 2, feature_range=(0, 9)):
    """Function to process the decoded Qwen Output into pairs of numbers, representing a two system time series. 

    Args:
        tokens (list): Tokens Outputted by Qwen's Prediction
        alphas (list): The alpha scaling factors used for pre-processing
        dp (int, optional): Number of decimal places. Defaults to 2.
        feature_range (tuple, optional): The min and max of the scaling range. Defaults to (0, 9).

    Raises:
        ValueError: Raised if the number of alphas and values are different

    Returns:
        List[List[float, float]]: List of post processed time series.
    """

    # Function to inverse the MinMaxScaling:
    def inverse_minmax_scale(X_scaled, scale_, feature_range=(0, 9)):
        min_range, max_range = feature_range

        # Calculate min_ from scale_ and feature_range
        # This assumes the original minimum value was at min_range after scaling
        data_range = max_range - min_range
        min_ = min_range - (min_range - max_range) / (scale_ * data_range)

        # Apply inverse transformation
        X_original = (X_scaled - min_) / scale_

        return X_original

    if len(tokens) != len(alphas):
        raise ValueError("The number of tokens and alphas must be the same.")

    # Empty array to hold the values
    predictions = []

    # Vocab of expected values
    vocab = [";", ",", ".", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]

    # Loop through each prediction and its respective alpha scaling factor
    for i, (token, alpha) in enumerate(zip(tokens, alphas)):

        # Filter out invalid strings using the vocab
        for char in token:
            if char not in vocab:
                token = token.replace(char, "")

        # Check if the string starts with a semi-colon:
        if token[0] == ";":
            token = token[1:]
        # Check if the string ends with a semi-colon:
        if token[-1] == ";":
            token = token[:-1]

        # Split into timesteps
        ts = token.split(";")

        # Create empty array to hold solutions
        system = [[], []]

        # Loop through each timestep and split into predator and prey
        for i in range(len(ts)):
            ind = ts[i].split(",")
            # Only add valid results (2 vals: prey and predator and matches the correct number of decimal places)
            if len(ind) == 2 and len(ind[0]) >= dp + 2 and len(ind[1]) >= dp + 2:
                system[0].append(
                    inverse_minmax_scale(float(ind[0]), alpha[0], feature_range)
                )
                system[1].append(
                    inverse_minmax_scale(float(ind[1]), alpha[1], feature_range)
                )

        # # Convert to numpy array
        # system = np.array(system)

        # # Rescale back to original scaling
        # print(system.shape)
        # system = alpha.inverse_transform(system)

        predictions.append(system)

    return predictions


# Splitting the data into input (X) and output (y):
def split_into_xy(data: list, n_pred: int = 20):
    """_summary_

    Args:
        data (list): List of post-processed time series
        n_pred (int, optional): Number of predictions. Defaults to 20.

    Returns:
        _type_: _description_
    """
    # Set up empty arrays
    x = []
    y = []
    for sequence in data:
        # Split the sequence of strings by ;
        steps = sequence.split(";")

        # Split into x and y
        left = steps[0 : len(steps) - n_pred - 1]
        right = steps[len(steps) - n_pred :]

        # Join the sequence together again
        x.append(";".join(left))
        y.append(";".join(right))

    return x, y, len(left), len(right)


def tokenize_data(data, tokenizer):
    """
    Tokenize the data for the Qwen model.

    Args:
        data: List of sequences as strings
        tokenizer: Qwen tokenizer

    Returns:
        List of tokenized sequences (as tensors)
    """
    # Tokenize each sequence
    tokenized = []
    for sequence in data:
        # Add any special formatting needed for your specific task
        encoded = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
        tokenized.append(encoded.input_ids.squeeze(0))

    return tokenized


def calculate_mse(val: list, pred: list):
    # Assert you have the same number of samples
    assert len(val) == len(pred)
    num_samples = len(val)

    mses = []

    for i in range(num_samples):
        # Index the pred and prey
        prey_val, pred_val = val[i]
        prey_pred, pred_pred = pred[i]

        # ensure the predictions and validation data have the same length
        min_len_prey = min(len(prey_val), len(prey_pred))
        min_len_pred = min(len(pred_val), len(pred_pred))

        # Calculate MSE for prey and predator separately
        mse_prey = mean_squared_error(prey_val[:min_len_prey], prey_pred[:min_len_prey])
        mse_pred = mean_squared_error(pred_val[:min_len_pred], pred_pred[:min_len_pred])

        # Average the MSE for prey and predator
        mses.append(mse_pred)
        mses.append(mse_prey)

    return mses

def calculate_mae(y_true, y_pred):
    maes = []
    for true_system, pred_system in zip(y_true, y_pred):
        if true_system is not None and pred_system is not None:
            # Calculate MAE for both prey and predator
            mae_prey = np.mean(np.abs(np.array(true_system[0][:min(len(true_system[0]), len(pred_system[0]))]) - np.array(pred_system[0][:min(len(true_system[0]), len(pred_system[0]))])))
            mae_pred = np.mean(np.abs(np.array(true_system[1][:min(len(true_system[1]), len(pred_system[1]))]) - np.array(pred_system[1][:min(len(true_system[1]), len(pred_system[1]))])))
            maes.append((mae_prey + mae_pred) / 2)
    return np.array(maes)
