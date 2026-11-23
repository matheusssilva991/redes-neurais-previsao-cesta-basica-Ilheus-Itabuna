"""Modelos e espaços de busca usados no experimento de previsão mensal."""

from collections.abc import Callable
from dataclasses import dataclass

import keras
import numpy as np
import pandas as pd
from keras import layers, ops
from sklearn.model_selection import ParameterGrid

from config import (
    CNN_FILTERS,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOOK_BACK,
    DEFAULT_NORMALIZATION_FACTOR,
    LSTM_UNITS_1,
    LSTM_UNITS_2,
    RNN_UNITS,
)
from src.models import get_model
from src.utils.data_utils import generate_forecast, prepare_training_data, train_model

PERIODO_SAZONAL = 12
TAMANHO_LOTE = 1
KERNEL_DLINEAR = 25

_PADRAO_TREINO = {
    "look_back": DEFAULT_LOOK_BACK,
    "learning_rate": DEFAULT_LEARNING_RATE,
    "epochs": DEFAULT_EPOCHS,
}
_PADROES_ATUAIS = {
    "RNN": {**_PADRAO_TREINO, "units": RNN_UNITS},
    "LSTM": {**_PADRAO_TREINO, "units_1": LSTM_UNITS_1, "units_2": LSTM_UNITS_2},
    "CNN": {**_PADRAO_TREINO, "filters": CNN_FILTERS},
}
_BUSCA_TREINO = {
    "look_back": [12, 24],
    "learning_rate": [0.0001, 0.0003, 0.001],
    "epochs": [50, 100, 150],
}
_BUSCAS = {
    "RNN": {**_BUSCA_TREINO, "units": [16, 24, 48]},
    "LSTM": {**_BUSCA_TREINO, "largura": [16, 32, 48]},
    "CNN": {**_BUSCA_TREINO, "filters": [16, 24, 48]},
    "DLinear": {**_BUSCA_TREINO, "kernel_size": [KERNEL_DLINEAR]},
}
_AJUSTES_CAMADAS = {
    "RNN": {"rnn_layer": ("units", "units")},
    "LSTM": {
        "lstm_layer_1": ("units", "units_1"),
        "lstm_layer_2": ("units", "units_2"),
    },
    "CNN": {"conv1d_layer": ("filters", "filters")},
}


@dataclass(frozen=True)
class Candidato:
    """Identifica um modelo, uma configuração experimental e seus parâmetros."""

    modelo: str
    nome: str
    parametros: dict[str, int | float | bool]


def gerar_candidatos(tentativas: int, semente: int, smoke: bool) -> list[Candidato]:
    """Gera os padrões atuais, as combinações sorteadas e as duas referências.

    Args:
        tentativas: Número máximo de combinações distintas por família.
        semente: Semente usada no sorteio sem reposição.
        smoke: Se verdadeiro, reduz o treino a uma época por candidata.

    Returns:
        Lista de candidatos em ordem fixa, com a configuração atual primeiro.
    """
    rng = np.random.default_rng(semente)
    candidatos = []
    for modelo, espaco in _BUSCAS.items():
        atual = _PADROES_ATUAIS.get(modelo)
        if atual is not None:
            candidatos.append(Candidato(modelo, "atual", atual.copy()))

        combinacoes = []
        for parametros in ParameterGrid(espaco):
            if modelo == "LSTM":
                largura = parametros.pop("largura")
                parametros.update(units_1=largura, units_2=largura)
            if parametros != atual:
                combinacoes.append(parametros)
        for numero, indice in enumerate(
            rng.permutation(len(combinacoes))[:tentativas], 1
        ):
            candidatos.append(
                Candidato(modelo, f"ajuste_{numero}", combinacoes[indice])
            )

    if smoke:
        for candidato in candidatos:
            candidato.parametros["epochs"] = 1

    return candidatos + [
        Candidato("Naive", "referencia", {}),
        Candidato("NaiveSazonal", "referencia", {"periodo": PERIODO_SAZONAL}),
    ]


def _montar_dlinear(look_back: int, horizonte: int, kernel_size: int) -> keras.Model:
    """Cria o DLinear univariado com extrapolação do valor nas bordas."""
    entrada = keras.Input(shape=(look_back, 1))
    margem = (kernel_size - 1) // 2
    estendida = ops.concatenate(
        [
            ops.repeat(entrada[:, :1], margem, axis=1),
            entrada,
            ops.repeat(entrada[:, -1:], margem, axis=1),
        ],
        axis=1,
    )
    tendencia = layers.AveragePooling1D(kernel_size, strides=1)(estendida)
    residuo = ops.subtract(entrada, tendencia)
    saida_tendencia = layers.Dense(horizonte)(layers.Flatten()(tendencia))
    saida_residuo = layers.Dense(horizonte)(layers.Flatten()(residuo))
    return keras.Model(
        entrada, layers.Add()([saida_tendencia, saida_residuo]), name="DLinear"
    )


def _montar_rede(candidato: Candidato, horizonte: int, semente: int) -> keras.Model:
    """Cria uma rede nova com a topologia atual ou a variante experimental."""
    keras.backend.clear_session()
    keras.utils.set_random_seed(semente)
    parametros = candidato.parametros
    look_back = int(parametros["look_back"])
    taxa = float(parametros["learning_rate"])
    if candidato.modelo == "DLinear":
        modelo = _montar_dlinear(look_back, horizonte, int(parametros["kernel_size"]))
    else:
        modelo_base = get_model(candidato.modelo, look_back, horizonte, taxa)
        if candidato.nome == "atual":
            return modelo_base

        ajustes = _AJUSTES_CAMADAS[candidato.modelo]

        def clonar_camada(camada: layers.Layer) -> layers.Layer:
            """Copia a camada com a largura experimental, quando aplicável."""
            configuracao = camada.get_config()
            if camada.name in ajustes:
                atributo, parametro = ajustes[camada.name]
                configuracao[atributo] = parametros[parametro]
            return type(camada).from_config(configuracao)

        keras.utils.set_random_seed(semente)
        modelo = keras.models.clone_model(modelo_base, clone_function=clonar_camada)

    modelo.compile(optimizer=keras.optimizers.Adam(taxa), loss="mean_squared_error")
    return modelo


def prever(
    historico: np.ndarray,
    horizonte: int,
    candidato: Candidato,
    semente: int,
    *,
    ao_fim_epoca: Callable[[int], None] | None = None,
) -> np.ndarray:
    """Ajusta o candidato com o prefixo observado e prevê um bloco em reais.

    Args:
        historico: Preços mensais disponíveis até a origem da previsão.
        horizonte: Número de meses consecutivos a prever.
        candidato: Modelo e parâmetros definidos antes de observar os alvos.
        semente: Semente do ajuste.
        ao_fim_epoca: Callback opcional que recebe o número da época concluída,
            a partir de 1, durante o treino de uma rede.

    Returns:
        Um valor para cada mês após a origem da previsão.
    """
    if candidato.modelo == "Naive":
        return np.repeat(historico[-1], horizonte)
    if candidato.modelo == "NaiveSazonal":
        return historico[-PERIODO_SAZONAL:][:horizonte].copy()

    parametros = candidato.parametros
    look_back = int(parametros["look_back"])
    entradas, alvos, ultima = prepare_training_data(
        pd.DataFrame({"preco": historico / DEFAULT_NORMALIZATION_FACTOR}),
        look_back,
        horizonte,
    )
    modelo = _montar_rede(candidato, horizonte, semente)
    callbacks = []
    if ao_fim_epoca is not None:
        callbacks.append(
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoca, _logs: ao_fim_epoca(epoca + 1)
            )
        )
    train_model(
        modelo,
        entradas,
        alvos,
        epochs=int(parametros["epochs"]),
        batch_size=TAMANHO_LOTE,
        callbacks=callbacks,
    )
    previsao = generate_forecast(modelo, ultima, batch_size=TAMANHO_LOTE)
    return np.asarray(previsao) * DEFAULT_NORMALIZATION_FACTOR
