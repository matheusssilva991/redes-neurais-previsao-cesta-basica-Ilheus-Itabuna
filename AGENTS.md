# AGENTS.md

## Projeto

O projeto prevê preços mensais da cesta básica e de seus produtos em Ilhéus e Itabuna. Consulte o [README](README.md) para os fluxos de atualização, treinamento e geração de gráficos.

- `config/` centraliza caminhos, regiões, produtos, arquiteturas e parâmetros padrão.
- `src/cli.py` e `src/main.ipynb` são as entradas de treinamento e previsão. A CLI usa o comando `previsao-cestas`.
- `src/ajustar_modelos.py` executa a busca temporal. A grade fica em `src/models/modelos_experimentais.py`.
- `src/utils/` reúne a leitura da planilha e de boletins, a preparação de dados, o treinamento e os gráficos.

## Ambiente e verificação

Use Python 3.12 com `uv`. O projeto usa Keras 3 com backend TensorFlow, conforme o `pyproject.toml` e o `uv.lock`. Configure o backend e a visibilidade de GPUs antes de importar Keras ao trabalhar nos pontos de entrada.

```bash
uv sync --locked --python 3.12
uv run --locked python -m unittest discover -s tests
uv run --locked python -m src.ajustar_modelos --smoke --saida /tmp/previsao-cestas-smoke
```

Use Ruff para verificar os arquivos Python alterados. O `--smoke` verifica o fluxo do ajuste em CPU com uma época por candidato. A busca completa com `--cpu` pode levar horas. Em testes de treino e exportação, use diretórios temporários para preservar os modelos, previsões e gráficos existentes.

## Dados e experimentos

- `data/precos_mensais.xlsx` é a fonte de dados. A chave é mês, cidade e produto. A cesta usa o primeiro dia do mês e os produtos usam o último dia válido.
- `importar-boletim` gera uma prévia por padrão. `--aplicar` persiste na planilha. `atualizar-precos` também grava na planilha. Preserve a planilha e o histórico em `previsoes_boletim/` em mudanças restritas ao código.
- Mantenha a ordem cronológica e use apenas o histórico disponível em cada origem. Selecione os parâmetros na validação e reserve o período de teste final para avaliação. Compare resultados usando a mesma planilha, séries, horizontes, cortes e semente.
- Os experimentos gravam manifesto, métricas, previsões e parâmetros em diretórios por execução em `output/experimentos_ajuste/`. As métricas experimentais são em reais. Preserve os hiperparâmetros do fluxo padrão ao alterar a grade experimental.

## Convenções de código e saída

- Reutilize os leitores e as funções de preparação existentes. Use docstrings em português no estilo Google, documentando argumentos, retorno e exceções relevantes. Use comentários curtos para decisões que afetam o comportamento.
- Preserve os nomes e formatos dos artefatos usados pelos gráficos. Os JSONs de previsão usam valores normalizados e o leitor aceita registros históricos. Use as funções existentes para converter valores em reais.
- Para exportação ONNX, preserve a entrada `float32` com forma `(1, look_back, 1)` e a paridade com o modelo `.keras` no ONNX Runtime. Cada chamada à LSTM exportada começa com estado zerado. A exportação usa uma cópia do modelo para preservar o estado do treinamento.
