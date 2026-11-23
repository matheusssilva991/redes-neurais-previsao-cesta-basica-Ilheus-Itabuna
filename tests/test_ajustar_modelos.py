import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from config import DEFAULT_EPOCHS
from src.ajustar_modelos import (
    SerieMensal,
    _avaliar,
    _cortes_temporais,
    _passos_por_corte,
    _selecionar,
)
from src.models.modelos_experimentais import Candidato, gerar_candidatos, prever


class AjusteModelosTest(unittest.TestCase):
    def setUp(self):
        self.serie = SerieMensal(
            "ilheus",
            "cesta_basica",
            pd.period_range("2015-01", periods=120, freq="M"),
            np.arange(120, dtype=float),
        )

    def test_cortes_avaliam_blocos_inteiros_com_o_historico_anterior(self):
        tamanho = len(self.serie.precos)
        for horizonte, n_testes in ((3, 8), (6, 4), (12, 2)):
            with self.subTest(horizonte=horizonte):
                validacao, teste = _cortes_temporais(tamanho, horizonte, 24, 4, 24)
                self.assertEqual(len(validacao), 4)
                self.assertEqual(validacao[-1].stop, tamanho - 24)
                self.assertEqual(len(teste), n_testes)
                self.assertEqual(teste[0].start, tamanho - 24)
                self.assertEqual(teste[-1].stop, tamanho)
                for etapa, blocos in (("validacao", validacao), ("teste", teste)):
                    with patch(
                        "src.models.modelos_experimentais.prever",
                        return_value=np.zeros(horizonte),
                    ) as ajustar:
                        linhas = _avaliar(
                            self.serie,
                            horizonte,
                            blocos,
                            Candidato("Naive", "referencia", {}),
                            etapa,
                            42,
                        )
                    for chamada, bloco in zip(
                        ajustar.call_args_list, blocos, strict=True
                    ):
                        np.testing.assert_array_equal(
                            chamada.args[0], self.serie.precos[: bloco.start]
                        )
                    self.assertEqual(
                        [linha["observado_reais"] for linha in linhas],
                        self.serie.precos[blocos[0].start : blocos[-1].stop].tolist(),
                    )
                    self.assertTrue(
                        all(linha["origem"] < linha["mes_previsto"] for linha in linhas)
                    )

    def test_selecao_usa_mae_de_validacao_e_prefere_atual_em_caso_de_empate(self):
        validacao, teste = _cortes_temporais(len(self.serie.precos), 3, 24, 4, 24)
        candidatos = [
            Candidato("RNN", "atual", {"erro": 2}),
            Candidato("RNN", "ajuste_1", {"erro": 1}),
            Candidato("RNN", "ajuste_2", {"erro": 3}),
            Candidato("LSTM", "ajuste_1", {"erro": 1}),
            Candidato("LSTM", "atual", {"erro": 1}),
            Candidato("Naive", "referencia", {"erro": 4}),
        ]

        def previsao_sintetica(
            historico, horizonte, candidato, semente, *, ao_fim_epoca=None
        ):
            return (
                historico[-1]
                + np.arange(1, horizonte + 1)
                + candidato.parametros["erro"]
            )

        with patch(
            "src.models.modelos_experimentais.prever", side_effect=previsao_sintetica
        ):
            linhas, melhores, final = _selecionar(
                self.serie, 3, validacao, candidatos, 42
            )

        self.assertTrue(all(linha["etapa"] == "validacao" for linha in linhas))
        self.assertLess(
            max(linha["mes_previsto"] for linha in linhas),
            str(self.serie.meses[teste[0].start]),
        )
        self.assertEqual(
            [
                (linha["modelo"], linha["candidato"], linha["mae_validacao_reais"])
                for linha in melhores
            ],
            [("RNN", "ajuste_1", 1.0), ("LSTM", "atual", 1.0)],
        )
        self.assertEqual(final, [candidatos[i] for i in (0, 4, 5, 1)])

    def test_progresso_conta_epocas_e_referencias_em_cada_corte(self):
        candidatos = [
            Candidato("RNN", "atual", {"epochs": 2}),
            Candidato("Naive", "referencia", {}),
        ]
        blocos = [slice(90, 93), slice(93, 96)]
        progresso = []

        def prever_sintetico(historico, horizonte, candidato, semente, *, ao_fim_epoca):
            for epoca in range(1, int(candidato.parametros.get("epochs", 0)) + 1):
                ao_fim_epoca(epoca)
            return np.repeat(historico[-1], horizonte)

        with patch("src.models.modelos_experimentais.prever", prever_sintetico):
            for candidato in candidatos:
                _avaliar(
                    self.serie,
                    3,
                    blocos,
                    candidato,
                    "teste",
                    42,
                    avancar=progresso.append,
                )

        self.assertEqual(len(progresso), len(blocos) * _passos_por_corte(candidatos))
        self.assertIn("RNN/atual corte 2/2 época 2/2", progresso[3])
        self.assertIn("Naive/referencia corte 2/2", progresso[-1])

    def test_referencia_sazonal_usa_os_mesmos_meses_do_ano_anterior(self):
        bloco = slice(83, 95)
        sazonal = Candidato("NaiveSazonal", "referencia", {"periodo": 12})
        resultado = prever(self.serie.precos[: bloco.start], 12, sazonal, 42)
        indices = self.serie.meses.get_indexer(self.serie.meses[bloco] - 12)
        np.testing.assert_array_equal(resultado, self.serie.precos[indices])

    def test_busca_e_reproduzivel_e_o_smoke_preserva_os_padroes(self):
        normal = gerar_candidatos(4, 42, False)
        smoke = gerar_candidatos(1, 42, True)
        self.assertEqual(normal, gerar_candidatos(4, 42, False))
        for grade in (normal, smoke):
            self.assertEqual(
                {candidato.modelo for candidato in grade},
                {"RNN", "LSTM", "CNN", "DLinear", "Naive", "NaiveSazonal"},
            )
        for modelo in ("RNN", "LSTM", "CNN"):
            candidatos = [c for c in normal if c.modelo == modelo]
            self.assertEqual(len(candidatos), 5)
            self.assertEqual(candidatos[0].nome, "atual")
            self.assertEqual(candidatos[0].parametros["epochs"], DEFAULT_EPOCHS)
        self.assertTrue(all(c.parametros.get("epochs", 1) == 1 for c in smoke))
