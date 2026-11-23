"""
Utility functions for preprocessing and training models.

Este módulo fornece funções para:
- Carregar e preprocessar séries da tabela única
- Criar sequências temporais para treinamento
- Preparar dados no formato Keras
- Treinar modelos
- Salvar modelos e previsões
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple, Optional, Union

import keras
import numpy as np
import onnx
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from keras.models import Sequential
from keras.callbacks import Callback, History

from config import (
    DEFAULT_NORMALIZATION_FACTOR,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    MODELS_DIR,
    FORECASTS_DIR,
    FORECASTS_PRODUTOS_DIR,
)
from .logger import setup_logger
from .monthly_data import PRODUCT_CESTA_BASICA, load_price_series

logger = setup_logger(__name__)


def load_data(
    path: Union[str, Path],
    normalize: bool = True,
    normalization_factor: int = DEFAULT_NORMALIZATION_FACTOR,
) -> pd.DataFrame:
    """Carrega preços de uma planilha Excel.

    Args:
        path: Caminho da planilha com a coluna `preco`.
        normalize: Se verdadeiro, divide os preços por `normalization_factor`.
        normalization_factor: Divisor aplicado a cada preço na normalização.

    Returns:
        DataFrame com a coluna `preco` na escala solicitada. A coluna `ano`
        é descartada quando presente.

    Raises:
        FileNotFoundError: Se a planilha não existir.
        ValueError: Se a planilha não puder ser lida ou não tiver a coluna `preco`.
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

    if "preco" not in df.columns:
        logger.error(f"Coluna 'preco' não encontrada em {path}")
        raise ValueError(f"Coluna 'preco' não encontrada no arquivo {path}")

    if "ano" in df.columns:
        df.drop(["ano"], axis=1, inplace=True)
        logger.debug("Coluna 'ano' removida")

    if normalize:
        df["preco"] = df["preco"] / normalization_factor
        logger.debug(f"Dados normalizados por fator {normalization_factor}")

    logger.info(f"Dados carregados com sucesso: {len(df)} registros")
    return df


def load_unified_data(
    city: str,
    product: str = PRODUCT_CESTA_BASICA,
    normalize: bool = True,
    normalization_factor: int = DEFAULT_NORMALIZATION_FACTOR,
) -> pd.DataFrame:
    """Carrega uma série mensal da planilha unificada de preços.

    Args:
        city: Cidade da série, como `ilheus` ou `itabuna`.
        product: Produto da série. Use `cesta_basica` para o total da cesta.
        normalize: Se verdadeiro, divide os preços por `normalization_factor`.
        normalization_factor: Divisor aplicado a cada preço na normalização.

    Returns:
        DataFrame ordenado por data, com as colunas `data` e `preco`.

    Raises:
        FileNotFoundError: Se a planilha unificada não existir.
        ValueError: Se a cidade ou o produto for inválido, a série estiver vazia
            ou a tabela mensal não puder ser normalizada.
    """
    logger.info(f"Carregando serie unificada: cidade={city}, produto={product}")
    df = load_price_series(city, product)

    if normalize:
        df = df.copy()
        df["preco"] = df["preco"] / normalization_factor
        logger.debug(f"Dados normalizados por fator {normalization_factor}")

    logger.info(f"Serie carregada com sucesso: {len(df)} registros")
    return df


def create_time_sequences(
    df: pd.DataFrame, look_back: int, forecast_horizon: int
) -> pd.DataFrame:
    """Cria janelas deslizantes de entrada e alvos a partir dos preços.

    Args:
        df: DataFrame com a coluna `preco` e, opcionalmente, a coluna `data`.
            Quando há datas, elas definem a ordem das observações.
        look_back: Número de observações passadas em cada janela de entrada.
        forecast_horizon: Número de observações futuras usadas como alvos.

    Returns:
        Uma linha por janela completa. As primeiras `look_back` colunas
        contêm as entradas e as demais contêm os alvos. Janelas com valores
        ausentes são descartadas.

    Raises:
        ValueError: Se faltar a coluna `preco` ou um dos tamanhos for menor que 1.

    Examples:
        >>> df = pd.DataFrame({"preco": [100, 110, 120, 130, 140]})
        >>> create_time_sequences(df, look_back=2, forecast_horizon=1).shape
        (3, 3)
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

    ordered_df = df.sort_values("data") if "data" in df.columns else df
    prices = ordered_df["preco"].reset_index(drop=True)
    window_size = look_back + forecast_horizon
    column_names = ["preco", *(f"preco t(h + {step})" for step in range(1, window_size))]
    columns = pd.Index(column_names, dtype=ordered_df.columns.dtype)
    if columns.hasnans:
        # Um índice categórico pode não ter os novos nomes entre suas categorias.
        columns = pd.Index(column_names)
    sequence_df = pd.concat(
        [prices.shift(-step) for step in range(window_size)],
        axis=1,
        keys=columns.rename(ordered_df.columns.name),
    ).dropna(ignore_index=True)

    logger.info(
        f"Sequências criadas: {len(sequence_df)} amostras, "
        f"{len(sequence_df.columns)} features"
    )
    return sequence_df


def _build_supervised_windows(
    prices: NDArray, look_back: int, forecast_horizon: int
) -> Tuple[NDArray, NDArray, NDArray]:
    """Monta os arrays de treino e a última janela de previsão.

    Args:
        prices: Array unidimensional de preços em ordem temporal.
        look_back: Quantidade de preços por janela de entrada.
        forecast_horizon: Quantidade de alvos após cada janela.

    Returns:
        Entradas de treino com forma `(n_amostras, look_back, 1)`, alvos
        com forma `(n_amostras, forecast_horizon)` e a última janela com
        forma `(1, look_back, 1)`. Os dois arrays de treino são float32.

    Raises:
        ValueError: Se a série tiver menos de `look_back + forecast_horizon`
            observações.
    """
    window_size = look_back + forecast_horizon
    if len(prices) < window_size:
        raise ValueError(
            f"Serie insuficiente: precisa de pelo menos {window_size} "
            f"observacoes, recebeu {len(prices)}"
        )

    windows = sliding_window_view(prices, window_size)
    # As janelas da view compartilham memória. O treino recebe cópias graváveis.
    X_train = windows[:, :look_back, None].astype(np.float32, order="C")
    y_train = windows[:, look_back:].astype(np.float32, order="C")
    X_val = prices[-look_back:].reshape(1, look_back, 1)
    return X_train, y_train, X_val


def prepare_training_data(
    df: pd.DataFrame, look_back: int, forecast_horizon: Optional[int] = None
) -> Tuple[NDArray, NDArray, NDArray]:
    """Prepara as entradas e os alvos do treino e a janela da próxima previsão.

    Args:
        df: Série bruta com a coluna `preco` e, opcionalmente, `data`, ou
            tabela de janelas de `create_time_sequences`, com entradas
            seguidas dos alvos em cada linha. Na série bruta, a coluna
            `data`, quando presente, define a ordem das observações.
        look_back: Número de observações por janela de entrada.
        forecast_horizon: Número de alvos por janela. Informe-o para séries
            brutas. Em janelas prontas, a omissão infere o horizonte pela
            quantidade de colunas de preços menos `look_back`.

    Returns:
        Tupla `(X_train, y_train, X_val)`. As formas são
        `(n_amostras, look_back, 1)`, `(n_amostras, forecast_horizon)` e
        `(1, look_back, 1)` quando há ao menos uma janela. `X_val` contém os
        últimos preços disponíveis para gerar a próxima previsão. Em uma
        tabela de janelas prontas vazia, os arrays têm zero amostras.

    Raises:
        ValueError: Se faltar `preco`, houver preços inválidos, os tamanhos
            forem inválidos, o horizonte não puder ser inferido ou a série
            bruta for curta demais para formar uma janela completa.
    """
    if look_back <= 0:
        raise ValueError(f"look_back deve ser > 0, recebido: {look_back}")

    if "preco" not in df.columns:
        raise ValueError("DataFrame deve conter coluna 'preco'")

    price_columns = [column for column in df.columns if column != "data"]

    if forecast_horizon is None:
        forecast_horizon = len(price_columns) - look_back
        if forecast_horizon <= 0:
            raise ValueError(
                "forecast_horizon precisa ser informado quando o DataFrame nao "
                "permite inferencia"
            )

    if forecast_horizon <= 0:
        raise ValueError(f"forecast_horizon deve ser > 0, recebido: {forecast_horizon}")

    logger.info(f"Preparando dados para treinamento: look_back={look_back}")

    if len(price_columns) >= look_back + forecast_horizon:
        supervised_df = df.loc[:, price_columns[: look_back + forecast_horizon]].apply(
            pd.to_numeric, errors="coerce"
        )
        if supervised_df.isna().any().any():
            raise ValueError("Colunas de preco contem valores invalidos")

        X_train = supervised_df.iloc[:, :look_back].values
        y_train = supervised_df.iloc[:, look_back : look_back + forecast_horizon].values
        X_val = supervised_df.iloc[-1:, -look_back:].values

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    else:
        ordered_df = df.sort_values("data") if "data" in df.columns else df
        prices = pd.to_numeric(ordered_df["preco"], errors="coerce")
        if prices.isna().any():
            raise ValueError("Coluna 'preco' contem valores invalidos")

        X_train, y_train, X_val = _build_supervised_windows(
            prices.to_numpy(dtype=np.float32), look_back, forecast_horizon
        )

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
    *,
    callbacks: list[Callback] | None = None,
) -> History:
    """Treina um modelo Keras mantendo a ordem das amostras.

    Args:
        model: Modelo Keras já compilado.
        X_train: Janelas de entrada na forma `(n_amostras, look_back, 1)`.
        y_train: Alvos na forma `(n_amostras, forecast_horizon)`.
        epochs: Número de épocas de treino.
        batch_size: Número de amostras por lote.
        verbose: Nível de saída do Keras. Use 0 para modo silencioso.
        callbacks: Callbacks acionados pelo Keras durante o treinamento.

    Returns:
        Histórico do Keras com a perda de treino por época.

    Raises:
        ValueError: Se `epochs` ou `batch_size` for menor que 1.
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
        callbacks=callbacks,
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
    save_onnx: bool = False,
) -> Path:
    """Grava um modelo Keras e, opcionalmente, o exporta para ONNX.

    Cria o diretório de saída e sobrescreve o arquivo Keras do mesmo
    modelo, região, série e horizonte. A exportação ONNX é opcional e suas
    falhas são registradas como avisos depois que o arquivo Keras é salvo.

    Args:
        model: Modelo Keras a serializar.
        model_name: Identificador da arquitetura, como `RNN`.
        region: Cidade que determina o diretório do modelo.
        object_name: Série prevista, como `cesta_basica` ou `arroz`.
        forecast_horizon: Horizonte em meses, incluído no nome do arquivo.
        subdir: Subdiretório opcional dentro da pasta da cidade.
        save_onnx: Se verdadeiro, tenta gravar também um arquivo `.onnx`.

    Returns:
        Caminho do arquivo `.keras`.

    Raises:
        OSError: Se houver erro de acesso ou escrita ao gravar o modelo Keras.
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
        logger.info("Modelo salvo com sucesso")
    except Exception as e:
        logger.error(f"Erro ao salvar modelo: {e}")
        raise

    if save_onnx:
        onnx_file = model_file.with_suffix(".onnx")
        try:
            logger.info(f"Exportando modelo ONNX em: {onnx_file}")
            _export_onnx(model_file, onnx_file)
            logger.info("Modelo ONNX salvo com sucesso")
        except Exception as e:
            logger.warning(f"Falha ao exportar ONNX ({onnx_file.name}): {e}")

    return model_file


def _export_onnx(model_file: Path, output_path: Path) -> None:
    """Exporta o modelo salvo para inferência ONNX em CPU.

    Args:
        model_file: Arquivo Keras com os pesos a exportar.
        output_path: Destino do arquivo ONNX validado.

    Raises:
        OSError: Se os arquivos não puderem ser lidos ou escritos.
        ValueError: Se a assinatura de entrada for incompatível com o modelo.
    """
    with keras.device("cpu"), TemporaryDirectory(dir=output_path.parent) as temp_dir:
        model = keras.saving.load_model(model_file, compile=False)
        if any(getattr(layer, "stateful", False) for layer in model.layers):
            # Cada chamada ONNX começa com estado zero. O modelo Keras
            # continua mantendo o estado entre lotes de treino e previsão.
            inference_model = keras.models.clone_model(
                model, clone_function=_clone_for_onnx
            )
            inference_model.set_weights(model.get_weights())
            model = inference_model
        signature = keras.InputSpec(
            shape=(1, *model.input_shape[1:]), dtype="float32", name="input"
        )
        temp_path = Path(temp_dir) / output_path.name
        model.export(
            str(temp_path), format="onnx", input_signature=[signature], verbose=False
        )
        onnx.checker.check_model(temp_path)
        temp_path.replace(output_path)


def _clone_for_onnx(layer: keras.layers.Layer) -> keras.layers.Layer:
    """Recria uma camada para chamadas ONNX independentes.

    Args:
        layer: Camada Keras a reconstruir.

    Returns:
        Camada com a mesma configuração, com estado recorrente desabilitado.
    """
    config = layer.get_config()
    if config.get("stateful"):
        config["stateful"] = False
    return layer.__class__.from_config(config)


def generate_forecast(
    model: Sequential, X_val: NDArray, batch_size: int = DEFAULT_BATCH_SIZE
) -> List[float]:
    """Gera a previsão correspondente à primeira janela de entrada.

    Args:
        model: Modelo Keras treinado.
        X_val: Janelas de inferência na forma `(n_amostras, look_back, 1)`,
            com pelo menos uma amostra.
        batch_size: Número de amostras por lote de inferência.

    Returns:
        Valores dos próximos passos, na escala de saída do modelo.

    Raises:
        ValueError: Se `batch_size` for menor que 1.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size deve ser > 0, recebido: {batch_size}")

    logger.info(f"Gerando previsões com batch_size={batch_size}")

    if len(X_val) == 1:
        # A chamada direta evita criar um grafo de predição para uma só janela.
        forecast = keras.ops.convert_to_numpy(model(X_val, training=False))
    else:
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
    """Grava em JSON os valores previstos para uma série.

    Cria os diretórios necessários e sobrescreve o arquivo se ele existir.
    Apenas o horizonte de 12 meses recebe a marca `12_meses` no nome.

    Args:
        results: Previsões na escala do modelo, armazenadas como um array JSON.
        model_name: Nome do modelo, como `RNN`, `LSTM` ou `CNN`.
        object_name: Nome da série, como `cesta_basica` ou `arroz`.
        region: Cidade da série, como `ilheus` ou `itabuna`.
        forecast_horizon: Horizonte da previsão em meses.
        forecast_type: Use `cesta` para a pasta de totais ou `produtos` para
            a pasta de produtos da cidade.

    Returns:
        Nome do arquivo JSON, sem o diretório.

    Raises:
        ValueError: Se `forecast_type` for inválido.
        OSError: Se ocorrer uma falha ao criar o diretório ou gravar o arquivo.
    """
    if forecast_type not in ["cesta", "produtos"]:
        raise ValueError(
            f"forecast_type deve ser 'cesta' ou 'produtos', recebido: {forecast_type}"
        )

    object_formatted = object_name.replace(" ", "_")

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
