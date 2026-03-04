"""
Utility functions for preprocessing and training models.

Este módulo fornece funções para:
- Carregar e preprocessar dados de Excel
- Criar sequências temporais para treinamento
- Preparar dados no formato Keras
- Treinar modelos
- Salvar modelos e previsões
"""

from typing import List, Tuple, Optional, Union
from pathlib import Path
import json
import os

import pandas as pd
import numpy as np
from numpy.typing import NDArray
from keras.models import Sequential
from keras.callbacks import History

from config import (
    DEFAULT_NORMALIZATION_FACTOR,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    MODELS_DIR,
    FORECASTS_DIR,
    FORECASTS_PRODUTOS_DIR,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


def load_data(
    path: Union[str, Path],
    normalize: bool = True,
    normalization_factor: int = DEFAULT_NORMALIZATION_FACTOR,
) -> pd.DataFrame:
    """
    Carrega e preprocessa dados de um arquivo Excel.

    Args:
        path (str | Path): Caminho para o arquivo Excel (.xlsx)
        normalize (bool, optional): Se True, normaliza os dados dividindo
                                   pelo fator de normalização. Padrão: True.
        normalization_factor (int, optional): Fator para normalização.
                                             Padrão: 1000.

    Returns:
        pd.DataFrame: DataFrame preprocessado com coluna 'preco'.
                     Se normalize=True, valores estarão divididos por 1000.

    Raises:
        FileNotFoundError: Se o arquivo não existir
        ValueError: Se o arquivo não puder ser lido ou não contiver
                   as colunas necessárias

    Examples:
        >>> df = load_data("../data/accb_custo_total_ilheus.xlsx")
        >>> df.head()
           mes    preco
        0  jan   0.550  # Normalizado (550 / 1000)
        1  fev   0.560

    Notes:
        - Remove automaticamente a coluna 'ano' se existir
        - Espera que o arquivo contenha coluna 'preco'
    """
    path = Path(path)

    if not path.exists():
        logger.error(f"Arquivo não encontrado: {path}")
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    try:
        logger.info(f"Carregando dados de: {path}")
        df = pd.read_excel(path)
    except Exception as e:
        logger.error(f"Erro ao ler arquivo {path}: {e}")
        raise ValueError(f"Erro ao ler arquivo {path}: {e}") from e

    # Validar que contém coluna preco
    if "preco" not in df.columns:
        logger.error(f"Coluna 'preco' não encontrada em {path}")
        raise ValueError(f"Coluna 'preco' não encontrada no arquivo {path}")

    # Remover coluna ano se existir
    if "ano" in df.columns:
        df.drop(["ano"], axis=1, inplace=True)
        logger.debug("Coluna 'ano' removida")

    if normalize:
        df["preco"] = df["preco"] / normalization_factor
        logger.debug(f"Dados normalizados por fator {normalization_factor}")

    logger.info(f"Dados carregados com sucesso: {len(df)} registros")
    return df


def create_time_sequences(
    df: pd.DataFrame, look_back: int, forecast_horizon: int
) -> pd.DataFrame:
    """
    Cria sequências temporais (sliding windows) para treinamento de séries temporais.

    Transforma série temporal em formato supervisionado, criando janelas deslizantes
    onde cada linha contém valores passados (features) e futuros (targets).

    Args:
        df (pd.DataFrame): DataFrame contendo coluna 'preco' com a série temporal
        look_back (int): Janela temporal de entrada (quantos períodos passados usar)
                        Deve ser > 0
        forecast_horizon (int): Quantos períodos futuros prever. Deve ser > 0

    Returns:
        pd.DataFrame: DataFrame com sequências temporais estruturadas.
                     Colunas: preco, preco t(h + 1), ..., preco t(h + n)
                     Linhas com NaN são removidas.

    Raises:
        ValueError: Se df não contém coluna 'preco' ou parâmetros inválidos

    Examples:
        >>> df = pd.DataFrame({'preco': [100, 110, 120, 130, 140]})
        >>> result = create_time_sequences(df, look_back=2, forecast_horizon=1)
        >>> result.shape
        (2, 3)  # 2 amostras, 3 colunas (preco + 2 shifts)

    Notes:
        - Usa shift negativo para criar colunas futuras
        - Remove automaticamente linhas com NaN (início/fim da série)
        - Reseta índice após remoção de NaN
    """
    if "preco" not in df.columns:
        logger.error("DataFrame não contém coluna 'preco'")
        raise ValueError("DataFrame deve conter coluna 'preco'")

    if look_back <= 0:
        raise ValueError(f"look_back deve ser > 0, recebido: {look_back}")
    if forecast_horizon <= 0:
        raise ValueError(f"forecast_horizon deve ser > 0, recebido: {forecast_horizon}")

    logger.info(
        f"Criando sequências temporais: look_back={look_back}, horizon={forecast_horizon}"
    )

    # Criar cópia para não modificar original
    df = df.copy()

    for n_step in range(1, look_back + forecast_horizon):
        df[f"preco t(h + {n_step})"] = df["preco"].shift(-n_step).values

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(f"Sequências criadas: {len(df)} amostras, {len(df.columns)} features")
    return df


def prepare_training_data(
    df: pd.DataFrame, look_back: int
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Prepara dados no formato Keras separando features (X) e targets (y).

    Divide o DataFrame em:
    - X_train: Janelas de entrada (look_back períodos)
    - y_train: Valores alvo (forecast_horizon períodos)
    - X_val: Última janela para validação (últimos 12 valores)

    Args:
        df (pd.DataFrame): DataFrame com sequências temporais criadas por
                          create_time_sequences()
        look_back (int): Tamanho da janela temporal de entrada

    Returns:
        Tuple[NDArray, NDArray, NDArray]: Tupla contendo:
            - X_train: shape (n_samples, look_back, 1)
            - y_train: shape (n_samples, forecast_horizon)
            - X_val: shape (1, 12, 1) para última predição

    Raises:
        ValueError: Se df não tiver colunas suficientes ou look_back inválido

    Examples:
        >>> df_sequences = create_time_sequences(df, look_back=3, forecast_horizon=1)
        >>> X_train, y_train, X_val = prepare_training_data(df_sequences, look_back=3)
        >>> X_train.shape
        (n_samples, 3, 1)

    Notes:
        - Reshape para formato Keras: (batches, timesteps, features)
        - X_val sempre usa últimos 12 valores (hardcoded para este projeto)
        - Assume que df já passou por create_time_sequences()
    """
    if look_back <= 0:
        raise ValueError(f"look_back deve ser > 0, recebido: {look_back}")

    if len(df.columns) < look_back:
        raise ValueError(
            f"DataFrame deve ter pelo menos {look_back} colunas, "
            f"tem apenas {len(df.columns)}"
        )

    logger.info(f"Preparando dados para treinamento: look_back={look_back}")

    X_train = df.iloc[:, :look_back].values
    y_train = df.iloc[:, look_back:].values
    X_val = df.iloc[-1:, -12:].values

    # Reshape to Keras format (batches, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

    logger.info(
        f"Dados preparados: X_train={X_train.shape}, "
        f"y_train={y_train.shape}, X_val={X_val.shape}"
    )

    return X_train, y_train, X_val


def train_model(
    model: Sequential,
    X_train: NDArray,
    y_train: NDArray,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    verbose: int = 0,
) -> History:
    """
    Treina um modelo Keras com os dados fornecidos.

    Args:
        model (Sequential): Modelo Keras compilado
        X_train (NDArray): Dados de entrada de treino.
                          Shape: (n_samples, timesteps, features)
        y_train (NDArray): Dados de saída de treino.
                          Shape: (n_samples, forecast_horizon)
        epochs (int, optional): Número de épocas de treinamento. Padrão: 150.
        batch_size (int, optional): Tamanho do batch. Padrão: 1.
        verbose (int, optional): Nível de verbosidade do Keras (0, 1, 2).
                               Padrão: 0 (silencioso).

    Returns:
        History: Objeto History do Keras com histórico de treinamento
                (loss por época, métricas, etc.)

    Raises:
        ValueError: Se epochs ou batch_size <= 0

    Examples:
        >>> model = get_model('RNN', look_back=12, forecast_horizon=3)
        >>> history = train_model(model, X_train, y_train, epochs=100)
        >>> history.history['loss'][-1]  # Loss final

    Notes:
        - shuffle=False para manter ordem temporal
        - Use verbose=1 para ver progresso durante treinamento
        - verbose=0 (padrão) é melhor para notebooks
    """
    if epochs <= 0:
        raise ValueError(f"epochs deve ser > 0, recebido: {epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size deve ser > 0, recebido: {batch_size}")

    logger.info(
        f"Iniciando treinamento: epochs={epochs}, batch_size={batch_size}, "
        f"samples={len(X_train)}"
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=verbose,
    )

    final_loss = history.history["loss"][-1]
    logger.info(f"Treinamento concluído. Loss final: {final_loss:.6f}")

    return history


def save_model(
    model: Sequential,
    model_name: str,
    region: str,
    object_name: str,
    forecast_horizon: int,
    subdir: Optional[str] = None,
) -> Path:
    """
    Salva um modelo Keras treinado em disco.

    O modelo é salvo no formato .keras (recomendado) no diretório de saída
    organizado por região e (opcionalmente) subdiretório.

    Args:
        model (Sequential): Modelo Keras treinado
        model_name (str): Nome do modelo (RNN, LSTM, CNN)
        region (str): Região ('ilheus' ou 'itabuna')
        object_name (str): Nome do objeto/produto (ex: 'cesta_basica', 'arroz')
        forecast_horizon (int): Horizonte de previsão (3, 6, 12, etc.)
        subdir (str, optional): Subdiretório adicional (ex: 'produtos').
                               Padrão: None.

    Returns:
        Path: Caminho completo do arquivo salvo

    Raises:
        OSError: Se não for possível criar diretórios ou salvar arquivo

    Examples:
        >>> model = get_model('RNN', 12, 3)
        >>> path = save_model(model, 'RNN', 'ilheus', 'cesta_basica', 3)
        >>> print(path)
        ../output/models/ilheus/RNN_ilheus_cesta_basica_h3.keras

        >>> # Com subdiretório
        >>> path = save_model(model, 'RNN', 'ilheus', 'arroz', 3, subdir='produtos')
        >>> print(path)
        ../output/models/ilheus/produtos/RNN_ilheus_arroz_h3.keras

    Notes:
        - Formato .keras é mais robusto que .h5
        - Cria diretórios automaticamente se não existirem
        - Sobrescreve arquivo se já existir
    """
    if subdir:
        save_dir = MODELS_DIR / region.lower() / subdir
    else:
        save_dir = MODELS_DIR / region.lower()

    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Erro ao criar diretório {save_dir}: {e}")
        raise

    model_file = (
        save_dir
        / f"{model_name}_{region.lower()}_{object_name.lower()}_h{forecast_horizon}.keras"
    )

    try:
        logger.info(f"Salvando modelo em: {model_file}")
        model.save(model_file)
        logger.info(f"Modelo salvo com sucesso")
    except Exception as e:
        logger.error(f"Erro ao salvar modelo: {e}")
        raise

    return model_file


def generate_forecast(
    model: Sequential, X_val: NDArray, batch_size: int = DEFAULT_BATCH_SIZE
) -> List[float]:
    """
    Gera previsões usando o modelo treinado.

    Args:
        model (Sequential): Modelo Keras treinado
        X_val (NDArray): Dados de validação/entrada para previsão.
                        Shape: (n_samples, timesteps, features)
        batch_size (int, optional): Tamanho do batch para previsão. Padrão: 1.

    Returns:
        List[float]: Lista de valores previstos (forecast_horizon valores)

    Raises:
        ValueError: Se batch_size <= 0

    Examples:
        >>> model = get_model('RNN', 12, 3)
        >>> # Treinar modelo...
        >>> predictions = generate_forecast(model, X_val)
        >>> len(predictions)
        3  # forecast_horizon = 3
        >>> predictions
        [0.551, 0.563, 0.572]  # Valores normalizados

    Notes:
        - O modelo deve estar treinado antes de chamar esta função
        - Retorna valores normalizados (se dados foram normalizados)
        - Use verbose=0 para suprimir mensagens do Keras
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size deve ser > 0, recebido: {batch_size}")

    logger.info(f"Gerando previsões com batch_size={batch_size}")

    forecast = model.predict(X_val, batch_size=batch_size, verbose=0)
    predictions = [float(value) for value in forecast[0]]

    logger.info(f"Previsões geradas: {len(predictions)} valores")
    logger.debug(f"Previsões: {predictions}")

    return predictions


def save_forecasts(
    results: List[float],
    model_name: str,
    object_name: str,
    region: str,
    forecast_horizon: int,
    forecast_type: str = "cesta",
) -> str:
    """
    Salva previsões em arquivo JSON.

    IMPORTANTE: Esta versão salva valores como array JSON nativo (não string),
    facilitando a leitura e interoperabilidade.

    Args:
        results (List[float]): Lista de valores previstos (normalizados)
        model_name (str): Nome do modelo (RNN, LSTM, CNN)
        object_name (str): Nome do objeto/produto (ex: 'cesta_basica', 'arroz')
        region (str): Região ('ilheus' ou 'itabuna')
        forecast_horizon (int): Horizonte de previsão usado
        forecast_type (str, optional): Tipo de previsão ('cesta' ou 'produtos').
                                      Padrão: 'cesta'.

    Returns:
        str: Nome do arquivo salvo (sem path completo)

    Raises:
        OSError: Se não for possível criar diretórios ou salvar arquivo
        ValueError: Se forecast_type inválido

    Examples:
        >>> results = [0.551, 0.563, 0.572]
        >>> filename = save_forecasts(
        ...     results, 'RNN', 'cesta_basica', 'ilheus', 3, 'cesta'
        ... )
        >>> print(filename)
        'previsao_RNN_cesta_basica_ilheus.json'

        >>> # Com horizonte de 12 meses
        >>> filename = save_forecasts(
        ...     results, 'RNN', 'arroz', 'ilheus', 12, 'produtos'
        ... )
        >>> print(filename)
        'previsao_RNN_12_meses_arroz_ilheus.json'

    Notes:
        - Valores são salvos como array JSON nativo: [0.551, 0.563, 0.572]
        - NÃO mais como string: "[0.551, 0.563, 0.572]"
        - Cria diretórios automaticamente se não existirem
        - Sobrescreve arquivo se já existir
        - ensure_ascii=False para suportar acentuação
    """
    if forecast_type not in ["cesta", "produtos"]:
        raise ValueError(
            f"forecast_type deve ser 'cesta' ou 'produtos', recebido: {forecast_type}"
        )

    object_formatted = object_name.replace(" ", "_")

    # IMPORTANTE: Salvar como array JSON nativo, não string
    output = {object_formatted.lower(): results}

    if forecast_horizon == 12:
        file_name = f"previsao_{model_name}_12_meses_{object_formatted.lower()}_{region.lower()}.json"
    else:
        file_name = (
            f"previsao_{model_name}_{object_formatted.lower()}_{region.lower()}.json"
        )

    if forecast_type == "cesta":
        full_path = FORECASTS_DIR / file_name
    else:
        full_path = FORECASTS_PRODUTOS_DIR / region / file_name

    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Salvando previsões em: {full_path}")

        with open(full_path, "w", encoding="utf-8") as file:
            json.dump(output, file, ensure_ascii=False, indent=2)

        logger.info(f"Previsões salvas com sucesso: {file_name}")

    except Exception as e:
        logger.error(f"Erro ao salvar previsões: {e}")
        raise

    return file_name
