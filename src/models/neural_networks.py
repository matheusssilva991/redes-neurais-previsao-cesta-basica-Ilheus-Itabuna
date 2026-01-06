"""
Definição das arquiteturas de redes neurais para previsão de séries temporais.
"""

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Input, LSTM, Conv1D, Flatten, MaxPooling1D
from keras.optimizers import Adam


def create_rnn_model(look_back, forecast_horizon, learning_rate=0.0003):
    """
    Rede Neural Recorrente (RNN) Simples

    Args:
        look_back: Janela temporal de entrada
        forecast_horizon: Número de períodos a prever
        learning_rate: Taxa de aprendizado do otimizador

    Returns:
        Sequential: Modelo RNN compilado
    """
    model = Sequential()
    model.add(Input(batch_shape=(1, look_back, 1)))
    model.add(SimpleRNN(24, stateful=False))
    model.add(Dense(forecast_horizon))
    model.compile(
        loss="mean_squared_error", optimizer=Adam(learning_rate=learning_rate) # type: ignore
    )
    return model


def create_lstm_model(look_back, forecast_horizon, learning_rate=0.0003):
    """
    Rede Neural Long Short-Term Memory (LSTM)

    Args:
        look_back: Janela temporal de entrada
        forecast_horizon: Número de períodos a prever
        learning_rate: Taxa de aprendizado do otimizador

    Returns:
        Sequential: Modelo LSTM compilado
    """
    model = Sequential()
    model.add(Input(batch_shape=(1, look_back, 1)))
    model.add(LSTM(32, stateful=False, return_sequences=True))
    model.add(LSTM(32, stateful=True))
    model.add(Dense(forecast_horizon))
    model.compile(
        loss="mean_squared_error", optimizer=Adam(learning_rate=learning_rate) # type: ignore
    )
    return model


def create_cnn_model(look_back, forecast_horizon, learning_rate=0.0003):
    """
    Rede Neural Convolucional 1D (CNN)

    Args:
        look_back: Janela temporal de entrada
        forecast_horizon: Número de períodos a prever
        learning_rate: Taxa de aprendizado do otimizador

    Returns:
        Sequential: Modelo CNN compilado
    """
    model = Sequential()
    model.add(Input(input_shape=(look_back, 1))) # type: ignore
    model.add(Conv1D(filters=24, kernel_size=3, activation="tanh"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(15, activation="tanh"))
    model.add(Dense(forecast_horizon))
    model.compile(
        loss="mean_squared_error", optimizer=Adam(learning_rate=learning_rate) # type: ignore
    )
    return model


# Dicionário de modelos disponíveis
AVAILABLE_MODELS = {
    "RNN": create_rnn_model,
    "LSTM": create_lstm_model,
    "CNN": create_cnn_model,
}


def get_model(model_name, look_back, forecast_horizon, learning_rate=0.0003):
    """
    Retorna um modelo compilado baseado no nome especificado.

    Args:
        model_name: Nome do modelo ('RNN', 'LSTM' ou 'CNN')
        look_back: Janela temporal de entrada
        forecast_horizon: Número de períodos a prever
        learning_rate: Taxa de aprendizado do otimizador

    Returns:
        Sequential: Modelo Keras compilado

    Raises:
        ValueError: Se o nome do modelo não for reconhecido
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Modelo '{model_name}' não encontrado. Modelos disponíveis: {list(AVAILABLE_MODELS.keys())}"
        )

    return AVAILABLE_MODELS[model_name](look_back, forecast_horizon, learning_rate)
