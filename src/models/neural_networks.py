"""
Definição das arquiteturas de redes neurais para previsão de séries temporais.

Este módulo implementa três arquiteturas de Deep Learning:
- RNN (Recurrent Neural Network): Rede neural recorrente simples
- LSTM (Long Short-Term Memory): Rede com memória de longo prazo
- CNN (Convolutional Neural Network): Rede convolucional 1D para séries temporais
"""

from typing import Callable, Dict
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Input, LSTM, Conv1D, Flatten, MaxPooling1D
from keras.optimizers import Adam

from config import (
    RNN_UNITS,
    LSTM_UNITS_1,
    LSTM_UNITS_2,
    CNN_FILTERS,
    CNN_KERNEL_SIZE,
    CNN_POOL_SIZE,
    CNN_DENSE_UNITS,
    DEFAULT_LEARNING_RATE,
)


def create_rnn_model(
    look_back: int, forecast_horizon: int, learning_rate: float = DEFAULT_LEARNING_RATE
) -> Sequential:
    """Cria uma RNN simples com saída linear para previsão de preços.

    A entrada tem uma variável por mês e lote fixo de uma amostra.
    A camada recorrente reinicia o estado a cada sequência.

    Args:
        look_back: Número de meses históricos na entrada.
        forecast_horizon: Número de valores futuros na saída.
        learning_rate: Taxa de aprendizado do otimizador Adam.

    Returns:
        Modelo Keras compilado com erro quadrático médio.

    Raises:
        ValueError: Se algum parâmetro for menor ou igual a zero.
    """
    if look_back <= 0:
        raise ValueError(f"look_back deve ser > 0, recebido: {look_back}")
    if forecast_horizon <= 0:
        raise ValueError(f"forecast_horizon deve ser > 0, recebido: {forecast_horizon}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate deve ser > 0, recebido: {learning_rate}")

    model = Sequential(name="RNN_Model")
    model.add(Input(batch_shape=(1, look_back, 1)))
    model.add(SimpleRNN(RNN_UNITS, stateful=False, name="rnn_layer"))
    model.add(Dense(forecast_horizon, name="output_layer"))
    model.compile(
        loss="mean_squared_error",
        optimizer=Adam(learning_rate=learning_rate),  # type: ignore
    )
    return model


def create_lstm_model(
    look_back: int, forecast_horizon: int, learning_rate: float = DEFAULT_LEARNING_RATE
) -> Sequential:
    """Cria uma rede com duas camadas LSTM e saída linear.

    A primeira camada entrega a sequência inteira à segunda, que mantém
    o estado entre lotes. A entrada usa uma variável por mês e lote fixo
    de uma amostra.

    Args:
        look_back: Número de meses históricos na entrada.
        forecast_horizon: Número de valores futuros na saída.
        learning_rate: Taxa de aprendizado do otimizador Adam.

    Returns:
        Modelo Keras compilado com erro quadrático médio.

    Raises:
        ValueError: Se algum parâmetro for menor ou igual a zero.
    """
    if look_back <= 0:
        raise ValueError(f"look_back deve ser > 0, recebido: {look_back}")
    if forecast_horizon <= 0:
        raise ValueError(f"forecast_horizon deve ser > 0, recebido: {forecast_horizon}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate deve ser > 0, recebido: {learning_rate}")

    model = Sequential(name="LSTM_Model")
    model.add(Input(batch_shape=(1, look_back, 1)))
    model.add(
        LSTM(LSTM_UNITS_1, stateful=False, return_sequences=True, name="lstm_layer_1")
    )
    model.add(LSTM(LSTM_UNITS_2, stateful=True, name="lstm_layer_2"))
    model.add(Dense(forecast_horizon, name="output_layer"))
    model.compile(
        loss="mean_squared_error",
        optimizer=Adam(learning_rate=learning_rate),  # type: ignore
    )
    return model


def create_cnn_model(
    look_back: int, forecast_horizon: int, learning_rate: float = DEFAULT_LEARNING_RATE
) -> Sequential:
    """Cria uma CNN unidimensional com saída linear para prever preços.

    A convolução e o pooling operam no eixo temporal, com uma variável
    de entrada por mês.

    Args:
        look_back: Número de meses históricos na entrada. Deve comportar
            o kernel da convolução e a janela do pooling configurados.
        forecast_horizon: Número de valores futuros na saída.
        learning_rate: Taxa de aprendizado do otimizador Adam.

    Returns:
        Modelo Keras compilado com erro quadrático médio.

    Raises:
        ValueError: Se algum parâmetro for menor ou igual a zero ou se a
            janela for incompatível com as dimensões das camadas.
    """
    if look_back <= 0:
        raise ValueError(f"look_back deve ser > 0, recebido: {look_back}")
    if forecast_horizon <= 0:
        raise ValueError(f"forecast_horizon deve ser > 0, recebido: {forecast_horizon}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate deve ser > 0, recebido: {learning_rate}")

    model = Sequential(name="CNN_Model")
    model.add(Input(shape=(look_back, 1)))
    model.add(
        Conv1D(
            filters=CNN_FILTERS,
            kernel_size=CNN_KERNEL_SIZE,
            activation="tanh",
            name="conv1d_layer",
        )
    )
    model.add(MaxPooling1D(pool_size=CNN_POOL_SIZE, name="maxpooling_layer"))
    model.add(Flatten(name="flatten_layer"))
    model.add(Dense(CNN_DENSE_UNITS, activation="tanh", name="dense_hidden"))
    model.add(Dense(forecast_horizon, name="output_layer"))
    model.compile(
        loss="mean_squared_error",
        optimizer=Adam(learning_rate=learning_rate),  # type: ignore
    )
    return model


# Dicionário de modelos disponíveis (factory pattern)
AVAILABLE_MODELS: Dict[str, Callable[[int, int, float], Sequential]] = {
    "RNN": create_rnn_model,
    "LSTM": create_lstm_model,
    "CNN": create_cnn_model,
}


def get_model(
    model_name: str,
    look_back: int,
    forecast_horizon: int,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> Sequential:
    """Cria e compila a arquitetura de previsão selecionada.

    Args:
        model_name: Nome exato da arquitetura, `RNN`, `LSTM` ou `CNN`.
        look_back: Número de meses históricos na entrada.
        forecast_horizon: Número de valores futuros na saída.
        learning_rate: Taxa de aprendizado do otimizador Adam.

    Returns:
        Modelo Keras compilado e pronto para treino.

    Raises:
        ValueError: Se a arquitetura for desconhecida ou se algum parâmetro
            numérico for inválido para o modelo selecionado.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Modelo '{model_name}' não encontrado. "
            f"Modelos disponíveis: {list(AVAILABLE_MODELS.keys())}"
        )

    return AVAILABLE_MODELS[model_name](look_back, forecast_horizon, learning_rate)
