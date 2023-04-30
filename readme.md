# Previsão do preço da cesta básica das cidades de Ilhéus e Itabuna

Neste projeto foram construídos três modelos de aprendizado profundo para a previsão de três meses do preço da cesta básica das cidades de Ilhéus e Itabuna. Os modelos desenvolvidos foram uma Rede Neural Recorrente (RNN), uma Rede Neural Convulacional (CNN) e uma Rede Neural *Long-Short Term Memory* (LSTM).

## Tabela de conteúdos

- [Estrutura das arquiteturas desenvolvidas](#estrutura-das-arquiteturas-desenvolvidas)
- [Configurar ambiente](#configurar-ambiente)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Quick Start](#quick-start)

## Estrutura das arquiteturas desenvolvidas

<div align="center">

**Rede Neural Recorrente**.

![Estrutura da Rede Neural Recorrente desenvolvida](img/RedeRNN.png)

**Rede Neural Long-Short Term Memory**.

![Estrutura da Rede Neural Recorrente desenvolvida](img/RedeLSTM.png)

**Rede Neural Convulacional**.

![Estrutura da Rede Neural Recorrente desenvolvida](img/RedeCNN.png)

</div>

## Configurar ambiente

- **Criar Ambiente**

    ```bash
    conda env create -f environment.yml
    ```

- **Ativar Ambiente**

    ```bash
    conda activate cesta
    ```

- **Desativar ambiente**

    ```bash
    conda deactivate cesta
    ```

## Estrutura do projeto

```text
 |--data\                           # Pasta que contém os conjuntos de dados das cidades
 |--img\                            # Imagens da estrutura do modelo
 |--ouput\                          # Arquivos de saída como os valores previstos e gráfico das previsões
 |--previsoes_boletim\              # Histórico das previsões feitas
 |--src\                            # Jupyter-nootebooks para executar os modelos e para gerar os graficos
 |--tests\                          # Arquivos de testes dos modelos
 |--configuracoes.conf              # Arquivo de configurações dos modelos
 |--configuracoes_graficos.conf     # Arquivo de configurações dos gráficos
```

## Quick Start

- **Entrar no jupyterlab**

    ```bash
    jupyter lab
    ```

- **Modificar arquivo configuracoes.conf**
  - Mudar cidade
  - Mudar diretório de trabalho
  - Mudar quantidade de Meses a serem previstos
  &nbsp;

- **Modificar arquivo configuracoes_grafico.config**

  - Selecionar modelo de previsão
  - Mudar meses anteriores e meses a serem previstos
    &nbsp;

- **Executar Notebooks modelos**

  - Executar jupyter-notebook model_RNN
  - Executar jupyter-notebook model_LSTM
  - Executar jupyter-notebook model_CNN
    &nbsp;

- **Ver Resultados/Graficos**

  - Executar jupyter-notebook grafico_cesta
  - Executar jupyter-notebook grafico_produtos
