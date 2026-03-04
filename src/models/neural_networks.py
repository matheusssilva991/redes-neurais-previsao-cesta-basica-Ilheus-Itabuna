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
    """
    Cria uma Rede Neural Recorrente (RNN) simples para previsão de séries temporais.

    A arquitetura consiste em uma camada SimpleRNN seguida de uma camada Dense
    para gerar as previsões. Adequada para capturar dependências de curto prazo.

    Args:
        look_back (int): Janela temporal de entrada (número de meses históricos).
                        Deve ser > 0.
        forecast_horizon (int): Número de períodos futuros a prever.
                               Deve ser > 0.
        learning_rate (float, optional): Taxa de aprendizado do otimizador Adam.
                                        Padrão: 0.0003.

    Returns:
        Sequential: Modelo RNN compilado e pronto para treinamento.
                   Loss: mean_squared_error, Optimizer: Adam.

    Raises:
        ValueError: Se look_back ou forecast_horizon <= 0

    Examples:
        >>> model = create_rnn_model(look_back=12, forecast_horizon=3)
        >>> model.summary()

    Notes:
        - Unidades RNN: 24 neurônios (configurável via RNN_UNITS)
        - Stateful: False (cada sequência é independente)
        - Função de ativação final: Linear (regressão)
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
    """
    Cria uma Rede Neural LSTM (Long Short-Term Memory) para séries temporais.

    Arquitetura com duas camadas LSTM empilhadas, capaz de aprender dependências
    de longo prazo nos dados. Ideal para séries temporais complexas.

    Args:
        look_back (int): Janela temporal de entrada (número de meses históricos).
                        Deve ser > 0.
        forecast_horizon (int): Número de períodos futuros a prever.
                               Deve ser > 0.
        learning_rate (float, optional): Taxa de aprendizado do otimizador Adam.
                                        Padrão: 0.0003.

    Returns:
        Sequential: Modelo LSTM compilado e pronto para treinamento.
                   Loss: mean_squared_error, Optimizer: Adam.

    Raises:
        ValueError: Se look_back ou forecast_horizon <= 0

    Examples:
        >>> model = create_lstm_model(look_back=12, forecast_horizon=3)
        >>> model.summary()

    Notes:
        - 1ª camada LSTM: 32 neurônios com return_sequences=True
        - 2ª camada LSTM: 32 neurônios com stateful=True
        - Função de ativação final: Linear (regressão)
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
    """
    Cria uma Rede Neural Convolucional 1D (CNN) para séries temporais.

    Aplica filtros convolucionais para extrair padrões temporais locais,
    seguido de pooling e camadas densas. Eficiente para detectar tendências.

    Args:
        look_back (int): Janela temporal de entrada (número de meses históricos).
                        Deve ser > 0.
        forecast_horizon (int): Número de períodos futuros a prever.
                               Deve ser > 0.
        learning_rate (float, optional): Taxa de aprendizado do otimizador Adam.
                                        Padrão: 0.0003.

    Returns:
        Sequential: Modelo CNN compilado e pronto para treinamento.
                   Loss: mean_squared_error, Optimizer: Adam.

    Raises:
        ValueError: Se look_back ou forecast_horizon <= 0

    Examples:
        >>> model = create_cnn_model(look_back=12, forecast_horizon=3)
        >>> model.summary()

    Notes:
        - Conv1D: 24 filtros, kernel_size=3, ativação tanh
        - MaxPooling1D: pool_size=2
        - Dense intermediária: 15 neurônios, ativação tanh
        - Função de ativação final: Linear (regressão)
    """
    if look_back <= 0:
        raise ValueError(f"look_back deve ser > 0, recebido: {look_back}")
    if forecast_horizon <= 0:
        raise ValueError(f"forecast_horizon deve ser > 0, recebido: {forecast_horizon}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate deve ser > 0, recebido: {learning_rate}")

    model = Sequential(name="CNN_Model")
    model.add(Input(input_shape=(look_back, 1)))  # type: ignore
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
    """
    Factory function para obter um modelo compilado pelo nome.

    Esta função implementa o padrão Factory para criar modelos de forma
    flexível e extensível. Novos modelos podem ser adicionados ao dicionário
    AVAILABLE_MODELS sem modificar esta função.

    Args:
        model_name (str): Nome do modelo. Opções: 'RNN', 'LSTM' ou 'CNN'.
                         Case-sensitive.
        look_back (int): Janela temporal de entrada (número de meses históricos).
                        Deve ser > 0.
        forecast_horizon (int): Número de períodos futuros a prever.
                               Deve ser > 0.
        learning_rate (float, optional): Taxa de aprendizado do otimizador Adam.
                                        Padrão: 0.0003.

    Returns:
        Sequential: Modelo Keras compilado e pronto para treinamento.

    Raises:
        ValueError: Se model_name não estiver em AVAILABLE_MODELS ou
                   se os parâmetros numéricos forem inválidos.

    Examples:
        >>> # Criar modelo RNN padrão
        >>> model = get_model('RNN', look_back=12, forecast_horizon=3)

        >>> # Criar modelo LSTM com learning rate customizado
        >>> model = get_model('LSTM', look_back=12, forecast_horizon=6, learning_rate=0.001)

        >>> # Listar modelos disponíveis
        >>> print(list(AVAILABLE_MODELS.keys()))
        ['RNN', 'LSTM', 'CNN']

    Notes:
        - Todos os modelos usam mean_squared_error como função de perda
        - Todos os modelos usam o otimizador Adam
        - Para adicionar novos modelos, basta incluir no AVAILABLE_MODELS
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Modelo '{model_name}' não encontrado. "
            f"Modelos disponíveis: {list(AVAILABLE_MODELS.keys())}"
        )

    return AVAILABLE_MODELS[model_name](look_back, forecast_horizon, learning_rate)
