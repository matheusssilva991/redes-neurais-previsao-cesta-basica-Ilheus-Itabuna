"""
Utility functions for preprocessing and training models.
"""

import pandas as pd
import json
import os
from pathlib import Path


def load_data(path, normalize=True, normalization_factor=1000):
    """
    Load and preprocess data from an Excel file.

    Args:
        path: Path to the Excel file
        normalize: Whether to normalize the data
        normalization_factor: Normalization factor

    Returns:
        DataFrame: Preprocessed data
    """
    df = pd.read_excel(path)
    df.drop(["ano"], axis=1, inplace=True)

    if normalize:
        df = df / normalization_factor

    return df


def create_time_sequences(df, look_back, forecast_horizon):
    """
    Create time sequences for time series training.

    Args:
        df: DataFrame with 'preco' column
        look_back: Input time window
        forecast_horizon: Number of periods to forecast

    Returns:
        DataFrame: DataFrame with time sequences
    """
    for n_step in range(1, look_back + forecast_horizon):
        df[f"preco t(h + {n_step})"] = df["preco"].shift(-n_step).values

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def prepare_training_data(df, look_back):
    """
    Prepare data for training by splitting into X and y.

    Args:
        df: DataFrame with time sequences
        look_back: Input time window

    Returns:
        tuple: (X_train, y_train, X_val)
    """
    X_train = df.iloc[:, :look_back].values
    y_train = df.iloc[:, look_back:].values
    X_val = df.iloc[-1:, -12:].values

    # Reshape to Keras format (batches, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

    return X_train, y_train, X_val


def train_model(model, X_train, y_train, epochs=150, batch_size=1, verbose=0):
    """
    Train a Keras model.

    Args:
        model: Keras model
        X_train: Training input data
        y_train: Training output data
        epochs: Number of epochs
        batch_size: Batch size
        verbose: Verbosity level

    Returns:
        History: Training history
    """
    return model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=verbose,
    )


def save_model(model, model_name, region, object_name, forecast_horizon, subdir=None):
    """
    Save a trained model.

    Args:
        model: Trained Keras model
        model_name: Model name (RNN, LSTM, CNN)
        region: Region (ilheus, itabuna)
        object_name: Object/product name
        forecast_horizon: Forecast horizon
        subdir: Additional subdirectory (e.g., 'produtos')

    Returns:
        Path: Saved file path
    """
    if subdir:
        save_dir = Path("../output/models") / region.lower() / subdir
    else:
        save_dir = Path("../output/models") / region.lower()

    save_dir.mkdir(parents=True, exist_ok=True)

    model_file = (
        save_dir
        / f"{model_name}_{region.lower()}_{object_name.lower()}_h{forecast_horizon}.keras"
    )
    model.save(model_file)

    return model_file


def generate_forecast(model, X_val, batch_size=1):
    """
    Generate forecasts using the trained model.

    Args:
        model: Trained Keras model
        X_val: Validation data
        batch_size: Batch size

    Returns:
        list: List of forecasts
    """
    forecast = model.predict(X_val, batch_size=batch_size, verbose=0)
    return [float(value) for value in forecast[0]]


def save_forecasts(
    results, model_name, object_name, region, forecast_horizon, forecast_type="cesta"
):
    """
    Save forecasts to JSON file.

    Args:
        results: List of forecasts
        model_name: Model name
        object_name: Object/product name
        region: Region
        forecast_horizon: Forecast horizon
        forecast_type: Forecast type ('cesta' or 'produtos')

    Returns:
        str: Saved file name
    """
    object_formatted = object_name.replace(" ", "_")
    output = {object_formatted.lower(): str(results)}

    if forecast_horizon == 12:
        file_name = f"previsao_{model_name}_12_meses_{object_formatted.lower()}_{region.lower()}.json"
    else:
        file_name = (
            f"previsao_{model_name}_{object_formatted.lower()}_{region.lower()}.json"
        )

    if forecast_type == "cesta":
        full_url = f"../output/previsoes_cesta/{file_name}"
    else:
        full_url = f"../output/previsoes_produtos/{region}/{file_name}"

    os.makedirs(os.path.dirname(full_url), exist_ok=True)

    with open(full_url, "w") as file:
        json.dump(output, file, ensure_ascii=False)

    return file_name
