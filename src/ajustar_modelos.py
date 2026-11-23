"""Ajusta e avalia modelos mensais em cortes cronológicos independentes."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import version
from io import BytesIO
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import click
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from config import (
    CNN_DENSE_UNITS,
    CNN_KERNEL_SIZE,
    CNN_POOL_SIZE,
    DEFAULT_NORMALIZATION_FACTOR,
    MASTER_DATA_FILE,
    OUTPUT_DIR,
    PRODUTOS,
    REGIOES,
)
from src.utils.monthly_data import PRODUCT_CESTA_BASICA, clean_master_dataframe

if TYPE_CHECKING:
    from src.models.modelos_experimentais import Candidato

HORIZONTES = (3, 6, 12)
MESES_TESTE = 24
CORTES_VALIDACAO = 4
MESES_SMOKE = 60
DIRETORIO_EXPERIMENTOS = OUTPUT_DIR / "experimentos_ajuste"


@dataclass(frozen=True)
class SerieMensal:
    """Preços em reais e meses consecutivos de um produto em uma cidade."""

    cidade: str
    produto: str
    meses: pd.PeriodIndex
    precos: np.ndarray


def _ler_series(
    planilha: bytes, cidades: tuple[str, ...], produtos: tuple[str, ...]
) -> list[SerieMensal]:
    """Lê as séries solicitadas da mesma cópia da planilha usada no manifesto."""
    tabela = clean_master_dataframe(pd.read_excel(BytesIO(planilha)))
    grupos = tabela.groupby(["cidade", "produto"])
    series = []
    for cidade, produto in product(cidades, produtos):
        if (cidade, produto) not in grupos.groups:
            raise ValueError(f"Série ausente na planilha: {cidade}/{produto}")

        dados = grupos.get_group((cidade, produto)).sort_values("data")
        meses = pd.PeriodIndex(pd.to_datetime(dados["data"]), freq="M")
        precos = dados["preco"].to_numpy(dtype=float)
        if not meses.equals(pd.period_range(meses[0], meses[-1], freq="M")):
            raise ValueError(
                f"A série precisa ter um registro por mês: {cidade}/{produto}"
            )
        if not np.isfinite(precos).all():
            raise ValueError(f"Preços inválidos na série: {cidade}/{produto}")

        series.append(SerieMensal(cidade, produto, meses, precos))
    return series


def _cortes_temporais(
    tamanho: int, horizonte: int, meses_teste: int, n_cortes: int, look_back_max: int
) -> tuple[list[slice], list[slice]]:
    """Define os blocos de avaliação sobre a série mensal, antes das janelas.

    O treino de cada corte é o prefixo anterior ao início do bloco.
    """
    from src.models.modelos_experimentais import PERIODO_SAZONAL

    if meses_teste % horizonte:
        raise ValueError(
            "O período de teste precisa conter blocos completos do horizonte."
        )
    desenvolvimento = tamanho - meses_teste
    primeiro_treino = desenvolvimento - n_cortes * horizonte
    minimo_treino = max(look_back_max + horizonte, PERIODO_SAZONAL)
    if primeiro_treino < minimo_treino:
        raise ValueError(
            f"Série com {tamanho} meses insuficiente para {n_cortes} cortes "
            f"de {horizonte} meses, treino mínimo de {minimo_treino} meses "
            f"e teste de {meses_teste} meses."
        )

    separador = TimeSeriesSplit(n_splits=n_cortes, test_size=horizonte)
    validacao = [
        slice(int(alvos[0]), int(alvos[-1]) + 1)
        for _, alvos in separador.split(range(desenvolvimento))
    ]
    teste = [
        slice(inicio, inicio + horizonte)
        for inicio in range(desenvolvimento, tamanho, horizonte)
    ]
    return validacao, teste


def _avaliar(
    serie: SerieMensal,
    horizonte: int,
    blocos: list[slice],
    candidato: Candidato,
    etapa: str,
    semente: int,
    *,
    avancar: Callable[[str], None] | None = None,
) -> list[dict]:
    """Ajusta novamente em cada origem e registra as previsões por mês.

    Args:
        serie: Série observada em ordem cronológica.
        horizonte: Número de meses em cada bloco.
        blocos: Cortes que indicam os meses a prever.
        candidato: Modelo e configuração a avaliar.
        etapa: Nome da etapa registrado nas previsões.
        semente: Semente usada em cada ajuste.
        avancar: Registra uma época concluída ou uma referência avaliada.

    Returns:
        Uma linha por mês previsto em cada corte.
    """
    from src.models.modelos_experimentais import prever

    linhas = []
    epocas = int(candidato.parametros.get("epochs", 0))
    for corte, bloco in enumerate(blocos, 1):
        historico = serie.precos[: bloco.start]
        origem = str(serie.meses[bloco.start - 1])
        rotulo = (
            f"{serie.cidade}/{serie.produto} H={horizonte} "
            f"{candidato.modelo}/{candidato.nome} corte {corte}/{len(blocos)}"
        )
        ao_fim_epoca = None
        if avancar is not None and epocas:

            def informar_epoca(epoca: int) -> None:
                avancar(f"{rotulo} época {epoca}/{epocas}")

            ao_fim_epoca = informar_epoca

        try:
            previsoes = prever(
                historico, horizonte, candidato, semente, ao_fim_epoca=ao_fim_epoca
            )
        except Exception as exc:
            raise RuntimeError(
                f"Falha em {serie.cidade}/{serie.produto}, H={horizonte}, "
                f"{candidato.modelo}/{candidato.nome}, {etapa}, origem {origem}: {exc}"
            ) from exc
        if previsoes.shape != (horizonte,) or not np.isfinite(previsoes).all():
            raise ValueError(
                f"Previsão inválida de {candidato.modelo}/{candidato.nome} em "
                f"{serie.cidade}/{serie.produto}, origem {origem}."
            )

        if avancar is not None and not epocas:
            avancar(rotulo)

        for passo, (mes, observado, previsto) in enumerate(
            zip(serie.meses[bloco], serie.precos[bloco], previsoes, strict=True), 1
        ):
            linhas.append(
                {
                    "etapa": etapa,
                    "cidade": serie.cidade,
                    "produto": serie.produto,
                    "horizonte": horizonte,
                    "modelo": candidato.modelo,
                    "candidato": candidato.nome,
                    "corte": corte,
                    "origem": origem,
                    "mes_previsto": str(mes),
                    "passo": passo,
                    "observado_reais": observado,
                    "previsto_reais": previsto,
                }
            )
    return linhas


def _selecionar(
    serie: SerieMensal,
    horizonte: int,
    blocos: list[slice],
    candidatos: list[Candidato],
    semente: int,
    *,
    avancar: Callable[[str], None] | None = None,
) -> tuple[list[dict], list[dict], list[Candidato]]:
    """Classifica as configurações usando apenas as previsões de validação."""
    linhas = []
    pontuacoes: dict[str, list[tuple[float, Candidato]]] = {}
    for candidato in candidatos:
        previsoes = _avaliar(
            serie, horizonte, blocos, candidato, "validacao", semente, avancar=avancar
        )
        linhas.extend(previsoes)
        if candidato.nome != "referencia":
            # Como todos os blocos têm o mesmo tamanho, cada corte tem peso igual.
            mae = float(
                mean_absolute_error(
                    [linha["observado_reais"] for linha in previsoes],
                    [linha["previsto_reais"] for linha in previsoes],
                )
            )
            pontuacoes.setdefault(candidato.modelo, []).append((mae, candidato))

    selecionados = []
    para_teste = [c for c in candidatos if c.nome in ("atual", "referencia")]
    for pontuacoes_modelo in pontuacoes.values():
        mae, candidato = min(
            pontuacoes_modelo,
            key=lambda resultado: (resultado[0], resultado[1].nome != "atual"),
        )
        mae_atual = next(
            (erro for erro, item in pontuacoes_modelo if item.nome == "atual"), None
        )
        selecionados.append(
            {
                "cidade": serie.cidade,
                "produto": serie.produto,
                "horizonte": horizonte,
                "modelo": candidato.modelo,
                "candidato": candidato.nome,
                "parametros": candidato.parametros,
                "mae_validacao_reais": mae,
                "mae_atual_validacao_reais": mae_atual,
            }
        )
        if candidato.nome != "atual":
            para_teste.append(candidato)
    return linhas, selecionados, para_teste


def _calcular_metricas(previsoes: pd.DataFrame) -> pd.DataFrame:
    """Resume MAE e RMSE em reais por corte, passo e conjunto de cortes."""
    grupo = ["etapa", "cidade", "produto", "horizonte", "modelo", "candidato"]
    erros = previsoes["previsto_reais"] - previsoes["observado_reais"]
    tabela = previsoes.assign(erro_absoluto=erros.abs(), erro_quadrado=erros.pow(2))
    resumos = []
    for nivel, chaves in (
        ("geral", []),
        ("corte", ["corte"]),
        ("passo", ["passo"]),
        ("corte_passo", ["corte", "passo"]),
    ):
        resumo = tabela.groupby(grupo + chaves, sort=False, as_index=False).agg(
            n_previsoes=("previsto_reais", "size"),
            mae_reais=("erro_absoluto", "mean"),
            mse=("erro_quadrado", "mean"),
        )
        resumo["rmse_reais"] = np.sqrt(resumo.pop("mse"))
        resumos.append(resumo.assign(nivel=nivel))
    resultado = pd.concat(resumos, ignore_index=True)
    resultado[["corte", "passo"]] = resultado[["corte", "passo"]].astype("Int64")
    return resultado[
        grupo + ["nivel", "corte", "passo", "n_previsoes", "mae_reais", "rmse_reais"]
    ]


def _gravar_json(caminho: Path, dados: object) -> None:
    """Grava o JSON e publica a versão completa com uma troca de arquivo."""
    temporario = caminho.with_suffix(".tmp")
    temporario.write_text(
        json.dumps(dados, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporario.replace(caminho)


def _passos_por_corte(candidatos: list[Candidato]) -> int:
    """Conta as épocas das redes e um passo por referência avaliada."""
    return sum(int(candidato.parametros.get("epochs", 1)) for candidato in candidatos)


def _barra_progresso(etapa: str, total: int, mostrar: bool):
    """Cria uma barra para a etapa quando o erro padrão vai para um terminal."""
    if not mostrar or not sys.stderr.isatty():
        return nullcontext()
    return click.progressbar(
        length=total,
        label=etapa,
        file=sys.stderr,
        show_pos=True,
        show_percent=True,
        item_show_func=lambda rotulo: rotulo,
    )


def executar(
    *,
    cidades: tuple[str, ...],
    produtos: tuple[str, ...],
    horizontes: tuple[int, ...],
    tentativas: int,
    semente: int,
    saida: Path = DIRETORIO_EXPERIMENTOS,
    planilha: Path = MASTER_DATA_FILE,
    smoke: bool = False,
    mostrar_progresso: bool = False,
) -> Path:
    """Executa a seleção cronológica e o teste final de um experimento.

    Args:
        cidades: Cidades a comparar.
        produtos: Séries alvo, incluindo o total da cesta quando solicitado.
        horizontes: Quantidades de meses a prever.
        tentativas: Máximo de candidatas sorteadas por família.
        semente: Semente da busca e do treino de cada candidato.
        saida: Diretório que recebe uma nova pasta de execução.
        planilha: Planilha unificada de preços.
        smoke: Se verdadeiro, usa as opções já reduzidas pelo modo de teste rápido.
        mostrar_progresso: Ativa as barras por época em um terminal interativo.

    Returns:
        Diretório com o manifesto, os parâmetros selecionados e as tabelas CSV.
    """
    import keras

    from src.models.modelos_experimentais import TAMANHO_LOTE, gerar_candidatos

    backend = keras.backend.backend()

    dados = planilha.read_bytes()
    series = _ler_series(dados, cidades, produtos)
    if smoke:
        series = [
            SerieMensal(
                s.cidade, s.produto, s.meses[-MESES_SMOKE:], s.precos[-MESES_SMOKE:]
            )
            for s in series
        ]
    n_cortes = 2 if smoke else CORTES_VALIDACAO
    meses_teste = horizontes[0] if smoke else MESES_TESTE
    candidatos = gerar_candidatos(tentativas, semente, smoke)
    look_back_max = max(int(c.parametros.get("look_back", 0)) for c in candidatos)
    casos = [
        (
            serie,
            horizonte,
            *_cortes_temporais(
                len(serie.precos), horizonte, meses_teste, n_cortes, look_back_max
            ),
        )
        for serie, horizonte in product(series, horizontes)
    ]
    inicio = datetime.now(timezone.utc)
    diretorio = saida / f"{inicio:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    diretorio.mkdir(parents=True)
    manifesto = {
        "status": "em_execucao",
        "inicio_utc": inicio.isoformat(),
        "smoke": smoke,
        "semente": semente,
        "tentativas_por_modelo": tentativas,
        "lote_keras": TAMANHO_LOTE,
        "backend_keras": backend,
        "dispositivo_keras": str(keras.ops.convert_to_tensor(0).device),
        "fator_normalizacao_keras": DEFAULT_NORMALIZATION_FACTOR,
        "cnn_parametros_fixos": {
            "kernel_size": CNN_KERNEL_SIZE,
            "pool_size": CNN_POOL_SIZE,
            "dense_units": CNN_DENSE_UNITS,
        },
        "versoes": {
            "python": platform.python_version(),
            **{
                pacote: version(pacote)
                for pacote in ("numpy", "pandas", "scikit-learn", "keras", backend)
            },
        },
        "planilha": str(planilha.resolve()),
        "planilha_sha256": hashlib.sha256(dados).hexdigest(),
        "avaliacoes": [
            {
                "cidade": s.cidade,
                "produto": s.produto,
                "horizonte": h,
                "inicio_serie": str(s.meses[0]),
                "fim_serie": str(s.meses[-1]),
                "origens_validacao": [str(s.meses[b.start - 1]) for b in cv],
                "origens_teste": [str(s.meses[b.start - 1]) for b in teste],
            }
            for s, h, cv, teste in casos
        ],
        "candidatos": [asdict(c) for c in candidatos],
    }
    arquivo_manifesto = diretorio / "manifesto.json"
    _gravar_json(arquivo_manifesto, manifesto)
    print(f"Resultados do experimento em {diretorio}", flush=True)

    linhas = []
    melhores = []
    teste_final = []
    try:
        total_validacao = _passos_por_corte(candidatos) * sum(
            len(validacao) for _, _, validacao, _ in casos
        )
        with _barra_progresso("Validação", total_validacao, mostrar_progresso) as barra:
            avancar = partial(barra.update, 1) if barra is not None else None
            for serie, horizonte, validacao, teste in casos:
                if barra is None:
                    print(
                        f"Validação de {serie.cidade}/{serie.produto}, H={horizonte}",
                        flush=True,
                    )
                previsoes, selecionados, para_teste = _selecionar(
                    serie, horizonte, validacao, candidatos, semente, avancar=avancar
                )
                linhas.extend(previsoes)
                melhores.extend(selecionados)
                teste_final.append((serie, horizonte, teste, para_teste))

        # Todas as escolhas são fechadas antes da primeira avaliação no teste.
        total_teste = sum(
            len(blocos) * _passos_por_corte(para_teste)
            for _, _, blocos, para_teste in teste_final
        )
        with _barra_progresso("Teste final", total_teste, mostrar_progresso) as barra:
            avancar = partial(barra.update, 1) if barra is not None else None
            for serie, horizonte, blocos, para_teste in teste_final:
                if barra is None:
                    print(
                        f"Teste de {serie.cidade}/{serie.produto}, H={horizonte}",
                        flush=True,
                    )
                for candidato in para_teste:
                    linhas.extend(
                        _avaliar(
                            serie,
                            horizonte,
                            blocos,
                            candidato,
                            "teste",
                            semente,
                            avancar=avancar,
                        )
                    )

        previsoes = pd.DataFrame(linhas)
        previsoes.to_csv(diretorio / "previsoes.csv", index=False)
        _calcular_metricas(previsoes).to_csv(diretorio / "metricas.csv", index=False)
        _gravar_json(diretorio / "melhores_parametros.json", melhores)
    except (Exception, KeyboardInterrupt) as exc:
        manifesto.update(status="falhou", erro=f"{type(exc).__name__}: {exc}")
        _gravar_json(arquivo_manifesto, manifesto)
        raise

    manifesto.update(status="concluido", fim_utc=datetime.now(timezone.utc).isoformat())
    _gravar_json(arquivo_manifesto, manifesto)
    return diretorio


def main() -> None:
    """Interpreta as opções do experimento e imprime o diretório de resultados."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cidade",
        action="append",
        choices=REGIOES,
        help="Cidade a avaliar. Repita para selecionar várias. Padrão: as duas.",
    )
    parser.add_argument(
        "--produto",
        action="append",
        choices=(PRODUCT_CESTA_BASICA, *PRODUTOS, "todos"),
        help="Série a avaliar. Repita ou use 'todos'. Padrão: cesta_basica.",
    )
    parser.add_argument(
        "--horizonte",
        action="append",
        type=int,
        choices=HORIZONTES,
        help="Meses a prever. Repita para selecionar vários. Padrão: 3 e 12.",
    )
    parser.add_argument(
        "--tentativas",
        type=int,
        default=4,
        help="Combinações por família até o limite do espaço de busca. Padrão: 4.",
    )
    parser.add_argument("--semente", type=int, default=42, help="Padrão: 42.")
    parser.add_argument(
        "--saida",
        type=Path,
        default=DIRETORIO_EXPERIMENTOS,
        help="Raiz dos relatórios experimentais. Padrão: output/experimentos_ajuste.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Valida o fluxo em CPU com Ilhéus, cesta básica e horizonte de 3 meses.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Executa o experimento em CPU com as GPUs NVIDIA desabilitadas.",
    )
    args = parser.parse_args()
    if args.tentativas < 1:
        parser.error("--tentativas deve ser maior que zero")
    if not 0 <= args.semente <= 2**32 - 1:
        parser.error("--semente deve estar entre 0 e 4294967295")

    cidades = tuple(dict.fromkeys(args.cidade or REGIOES))
    produtos = tuple(dict.fromkeys(args.produto or [PRODUCT_CESTA_BASICA]))
    if "todos" in produtos:
        produtos = (PRODUCT_CESTA_BASICA, *PRODUTOS)
    horizontes = tuple(dict.fromkeys(args.horizonte or (3, 12)))
    tentativas = args.tentativas
    if args.cpu or args.smoke:
        # O backend consulta essa variável ao inicializar os dispositivos.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("Execução em CPU.", flush=True)

    if args.smoke:
        cidades, produtos, horizontes, tentativas = (
            ("ilheus",),
            (PRODUCT_CESTA_BASICA,),
            (3,),
            1,
        )

    from src.utils.data_utils import logger as logger_dados

    logger_dados.setLevel(logging.WARNING)
    try:
        resultado = executar(
            cidades=cidades,
            produtos=produtos,
            horizontes=horizontes,
            tentativas=tentativas,
            semente=args.semente,
            saida=args.saida,
            smoke=args.smoke,
            mostrar_progresso=True,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Erro: {exc}\n")
    print(f"Experimento concluído: {resultado}")


if __name__ == "__main__":
    main()
