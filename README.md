# 🛒 Previsão do Preço da Cesta Básica - Ilhéus e Itabuna

Sistema de previsão de preços utilizando Deep Learning para análise e projeção dos valores da cesta básica e seus produtos individuais nas cidades de Ilhéus e Itabuna, Bahia.

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Arquiteturas de Redes Neurais](#arquiteturas-de-redes-neurais)
- [Arquitetura do Código](#arquitetura-do-código)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Configuração do Ambiente](#configuração-do-ambiente)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Como Usar](#como-usar)
- [Configurações](#configurações)
- [Resultados e Visualizações](#resultados-e-visualizações)
- [Produtos Analisados](#produtos-analisados)
- [Contribuindo](#contribuindo)

## 🎯 Sobre o Projeto

Este projeto implementa três modelos de aprendizado profundo para prever o preço da cesta básica e seus produtos individuais nas cidades de Ilhéus e Itabuna, Bahia. O sistema utiliza séries temporais históricas de preços para realizar previsões de curto e médio prazo (3 a 12 meses).

### Características Principais

- ✅ Previsão do valor total da cesta básica
- ✅ Previsão individualizada para 12 produtos
- ✅ Três arquiteturas de redes neurais implementadas (RNN, LSTM, CNN)
- ✅ Análise comparativa entre duas cidades
- ✅ Geração automatizada de gráficos e relatórios
- ✅ Histórico completo de previsões anteriores

### Modelos Implementados

1. **RNN (Recurrent Neural Network)** - Rede Neural Recorrente simples
2. **LSTM (Long Short-Term Memory)** - Rede com memória de longo prazo
3. **CNN (Convolutional Neural Network)** - Rede Neural Convolucional 1D

## 🧠 Arquiteturas de Redes Neurais

### Rede Neural Recorrente (RNN)

A RNN básica processa sequências temporais mantendo informações de estados anteriores.

<div align="center">

![Estrutura da Rede Neural Recorrente desenvolvida](img/RedeRNN.png)

**Arquitetura:** Input(12 meses) → SimpleRNN(24 neurônios) → Dense(n_previsões)

</div>

### Rede Neural LSTM

A LSTM resolve o problema de dependências de longo prazo através de gates de memória.

<div align="center">

![Estrutura da Rede Neural LSTM desenvolvida](img/RedeLSTM.png)

**Arquitetura:** Input(12 meses) → LSTM(32) → LSTM(32) → Dense(n_previsões)

</div>

### Rede Neural Convolucional (CNN)

A CNN 1D aplica filtros convolucionais para extrair padrões temporais.

<div align="center">

![Estrutura da Rede Neural CNN desenvolvida](img/RedeCNN.png)

**Arquitetura:** Input(12 meses) → Conv1D(24 filtros) → MaxPooling1D → Flatten → Dense(15) → Dense(n_previsões)

</div>

## 🏗️ Arquitetura do Código

O projeto foi estruturado seguindo boas práticas de engenharia de software, com código modular e reutilizável.

### Pacotes Principais

#### 📦 `src/models/` - Definições de Modelos

Contém as arquiteturas das redes neurais em um módulo centralizado:

```python
from models import get_model

# Obter modelo configurado
model = get_model('RNN', look_back=12, forecast_horizon=3)
model = get_model('LSTM', look_back=12, forecast_horizon=3)
model = get_model('CNN', look_back=12, forecast_horizon=3)
```

**Arquivos:**

- `neural_networks.py` - Implementações de RNN, LSTM e CNN
- `__init__.py` - Exporta função factory `get_model()`

#### 📦 `src/utils/` - Utilitários

Funções reutilizáveis para processamento de dados e visualização:

**`data_utils.py`** - Processamento e Treinamento

```python
from utils import (
    load_data,              # Carrega dados do Excel
    create_time_sequences,  # Cria sequências temporais
    prepare_training_data,  # Prepara dados para treinamento
    train_model,            # Treina o modelo
    generate_forecast,      # Gera previsões
    save_forecasts          # Salva resultados em JSON
)
```

**`chart_utils.py`** - Visualização

```python
from utils import (
    load_forecasts_for_regions,     # Carrega previsões de regiões
    load_product_forecasts,         # Carrega previsões de produtos
    load_product_historical_data,   # Carrega dados históricos
    setup_plot_style,               # Configura estilo matplotlib
    format_yticks_with_comma,       # Formata eixo Y (vírgula decimal)
    plot_product_chart,             # Plota gráfico de produtos
    plot_forecast_only_chart,       # Plota apenas previsões
    save_figure                     # Salva figura PNG
)
```

### Notebooks Otimizados

Todos os notebooks foram refatorados para usar os pacotes `models/` e `utils/`:

- ✅ **Código sem duplicação** - Funções reutilizadas entre notebooks
- ✅ **Nomenclatura em inglês** - Padrão internacional para funções
- ✅ **Configurações globais** - Parâmetros centralizados no início
- ✅ **Modularidade** - Fácil manutenção e expansão

## 🛠️ Tecnologias Utilizadas

### Principais Bibliotecas

- **Python 3.12** - Linguagem de programação
- **TensorFlow/Keras** - Framework de Deep Learning
- **Pandas** - Manipulação de dados
- **Matplotlib** - Visualização de gráficos
- **Seaborn** - Visualizações estatísticas
- **Scikit-learn** - Pré-processamento e métricas
- **openpyxl** - Leitura de arquivos Excel

## ⚙️ Configuração do Ambiente

### Pré-requisitos

- Conda ou Miniconda instalado
- Python 3.12 ou superior
- Git (opcional)

### Instalação

1. **Clone o repositório**

    ```bash
    git clone https://github.com/matheusssilva991/redes-neurais-previsao-cesta-basica-Ilheus-Itabuna.git
    cd redes-neurais-previsao-cesta-basica-Ilheus-Itabuna
    ```

2. **Crie e ative o ambiente Conda**

    ```bash
    # Criar ambiente a partir do arquivo environment.yml
    conda env create -f environment.yml

    # Ativar o ambiente
    conda activate cesta
    ```

3. **Ou usando uv (alternativa moderna)**

    ```bash
    # Instalar dependências via pyproject.toml
    uv sync
    ```

4. **Desativar ambiente (quando necessário)**

    ```bash
    conda deactivate
    ```

## 📁 Estrutura do Projeto

```text
previsao_cestas/
├── data/                                    # Dados de entrada
│   └── datasets_produtos/                   # Datasets por produto e cidade
│       ├── ilheus/                          # Dados de Ilhéus
│       └── itabuna/                         # Dados de Itabuna
├── img/                                     # Imagens das arquiteturas
│   ├── RedeRNN.png
│   ├── RedeLSTM.png
│   └── RedeCNN.png
├── output/                                  # Resultados gerados
│   ├── figure/                              # Gráficos de visualização
│   │   ├── produtos_ilheus/                 # Gráficos dos produtos de Ilhéus
│   │   └── produtos_itabuna/                # Gráficos dos produtos de Itabuna
│   ├── models/                              # Modelos treinados (.keras)
│   │   ├── ilheus/                          # Modelos de Ilhéus
│   │   │   └── produtos/                    # Modelos por produto
│   │   └── itabuna/                         # Modelos de Itabuna
│   │       └── produtos/                    # Modelos por produto
│   ├── previsoes_cesta/                     # Previsões da cesta completa (JSON)
│   └── previsoes_produtos/                  # Previsões por produto (JSON)
│       ├── ilheus/
│       └── itabuna/
├── previsoes_boletim/                       # Histórico de previsões por período
│   ├── 2022/, 2023/, 2024/, 2025/          # Organizados por ano e mês
│   └── previsao_XXXX_completo/             # Previsões anuais completas
├── src/                                     # Código fonte
│   ├── main.ipynb                           # Notebook principal de treinamento
│   ├── models/                              # 📦 Pacote de modelos
│   │   ├── __init__.py                      # Exporta get_model()
│   │   └── neural_networks.py               # Arquiteturas RNN, LSTM, CNN
│   ├── utils/                               # 📦 Pacote de utilitários
│   │   ├── __init__.py                      # Exporta todas as funções
│   │   ├── data_utils.py                    # Processamento e treinamento
│   │   ├── chart_utils.py                   # Visualização de gráficos
│   │   └── README.md                        # Documentação das funções
│   └── graficos/                            # Notebooks de visualização
│       ├── graficos_3_meses.ipynb           # Gráficos de 3 meses (otimizado)
│       ├── graficos_12_meses.ipynb          # Gráficos de 12 meses (otimizado)
│       └── graficos_produtos.ipynb          # Gráficos por produto (legado)
├── tests/                                   # Testes e avaliações
│   ├── avaliar_modelos_cv.ipynb             # Avaliação com validação cruzada
│   └── avaliar_modelos_ultimos_3_meses_2021.ipynb
├── configuracoes_graficos.conf              # Configurações dos gráficos
├── environment.yml                          # Dependências Conda
├── pyproject.toml                           # Configuração do projeto (uv/pip)
└── README.md                                # Este arquivo
```

## 🚀 Como Usar

### 1. Treinamento dos Modelos

O notebook principal foi otimizado com **variáveis globais** e funções reutilizáveis dos pacotes `models/` e `utils/`.

#### Configurar Parâmetros Globais

Abra [src/main.ipynb](src/main.ipynb) e ajuste as variáveis na primeira célula:

```python
# Configurações globais
MODEL_NAME = 'RNN'          # Opções: 'RNN', 'LSTM', 'CNN'
FORECAST_HORIZON = 3        # Meses a prever: 3, 6 ou 12
LOOK_BACK = 12              # Janela de observação (meses)
EPOCHS = 150                # Épocas de treinamento
BATCH_SIZE = 1              # Tamanho do batch
```

#### Executar Treinamento

1. Abra o Jupyter Lab:

   ```bash
   jupyter lab
   ```

2. Navegue até `src/` e abra [main.ipynb](src/main.ipynb)

3. Execute todas as células para:
   - Carregar dados automaticamente
   - Treinar modelo selecionado usando `models.get_model()`
   - Gerar previsões com `utils.generate_forecast()`
   - Salvar modelos em `.keras` com `utils.save_model()`
   - Exportar previsões em JSON com `utils.save_forecasts()`

**O notebook processa automaticamente:**

- ✅ Previsão da cesta básica completa (ambas as cidades)
- ✅ Previsões individuais dos 12 produtos (ambas as cidades)
- ✅ Total: 26 modelos treinados por execução

### 2. Visualização dos Resultados

Os notebooks de gráficos foram **completamente otimizados** usando funções do pacote `utils/`.

#### Gráficos de 3 Meses (com histórico)

**[src/graficos/graficos_3_meses.ipynb](src/graficos/graficos_3_meses.ipynb)**

Exibe valores históricos (9 meses) + previsões (3 meses):

```python
# Configuração automática de yticks
# Ajusta intervalo baseado no range (evita excesso de marcações)

# Gera automaticamente:
# - Gráfico da cesta básica (ambas as cidades)
# - Gráficos de produtos de Ilhéus
# - Gráficos de produtos de Itabuna
```

**Características:**

- 📊 Valores reais em linhas sólidas
- 📈 Previsões em linhas pontilhadas
- 🎯 Anotações automáticas de valores
- 📐 Ajuste inteligente de escala do eixo Y

#### Gráficos de 12 Meses (apenas previsões)

**[src/graficos/graficos_12_meses.ipynb](src/graficos/graficos_12_meses.ipynb)**

Exibe apenas as previsões para o ano completo:

```python
# Usa função otimizada: plot_forecast_only_chart()

# Gera automaticamente:
# - Gráfico da cesta básica (ano completo)
# - Produtos de Ilhéus (12 meses)
# - Produtos de Itabuna (12 meses)
```

#### Configurações dos Gráficos

Edite [configuracoes_graficos.conf](configuracoes_graficos.conf) para ajustar:

```ini
[CESTA_BASICA]
qtd_meses_previstos = 3
meses_previstos = outubro, novembro, dezembro
meses_anteriores = janeiro, fevereiro, ..., setembro
valores_reais_ilheus = [650.5, 655.3, ..., 690.1]
valores_reais_itabuna = [645.2, 650.1, ..., 685.5]
eixo_y_limite_sup = 750
eixo_y_limite_inf = 600
ano_previsao = 2025
modelo_atual = RNN

[PRODUTOS]
pasta = ../../output
subpasta_ios = previsoes_produtos/ilheus
subpasta_itb = previsoes_produtos/itabuna
subpasta_valores_reais_ios = datasets_produtos/ilheus
subpasta_valores_reais_itb = datasets_produtos/itabuna
eixo_y_limite_sup = 80
```

### 3. Formato dos Dados de Entrada

Os arquivos Excel devem conter:

| ano  | preco  |
|------|--------|
| 2020 | 650.50 |
| 2020 | 655.30 |
| ...  | ...    |

**Colunas necessárias:**

- `ano` - Ano da observação
- `preco` - Valor em Reais

## ⚙️ Configurações

### Hiperparâmetros dos Modelos

Os modelos são configurados via variáveis globais em [main.ipynb](src/main.ipynb):

```python
# Parâmetros de treinamento
MODEL_NAME = 'RNN'          # Modelo: RNN, LSTM ou CNN
FORECAST_HORIZON = 3        # Horizonte: 3, 6 ou 12 meses
LOOK_BACK = 12              # Janela temporal: 12 meses
EPOCHS = 150                # Épocas de treinamento
BATCH_SIZE = 1              # Tamanho do batch
LEARNING_RATE = 0.0003      # Taxa de aprendizado (Adam optimizer)
```

### Arquiteturas dos Modelos

Definidas no módulo [src/models/neural_networks.py](src/models/neural_networks.py):

**RNN:**

```python
def create_rnn_model(look_back, forecast_horizon):
    model = Sequential([
        SimpleRNN(24, input_shape=(look_back, 1)),
        Dense(forecast_horizon)
    ])
    return model
```

**LSTM:**

```python
def create_lstm_model(look_back, forecast_horizon):
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(look_back, 1)),
        LSTM(32, return_sequences=False),
        Dense(forecast_horizon)
    ])
    return model
```

**CNN:**

```python
def create_cnn_model(look_back, forecast_horizon):
    model = Sequential([
        Conv1D(24, kernel_size=3, activation='relu', input_shape=(look_back, 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(15, activation='relu'),
        Dense(forecast_horizon)
    ])
    return model
```

### Funções Utilitárias

Documentação completa disponível em [src/utils/README.md](src/utils/README.md).

**Principais funções:**

- `load_data()` - Carrega e valida dados do Excel
- `create_time_sequences()` - Cria janelas temporais
- `prepare_training_data()` - Normaliza e formata dados
- `train_model()` - Treina com early stopping
- `generate_forecast()` - Gera previsões futuras
- `plot_product_chart()` - Plota com histórico + previsão
- `plot_forecast_only_chart()` - Plota apenas previsão

### Personalização Avançada

Para modificar arquiteturas:

1. Edite `src/models/neural_networks.py`
2. Adicione novos modelos ou ajuste camadas existentes
3. A função `get_model()` detecta automaticamente as mudanças

## 📊 Resultados e Visualizações

### Formatos de Saída

1. **Modelos Treinados** (.keras)
   - Localização: `output/models/{cidade}/`
   - Formato: `{MODELO}_{cidade}_{objeto}_h{horizonte}.keras`

2. **Previsões** (JSON)
   - Localização: `output/previsoes_cesta/` ou `output/previsoes_produtos/`
   - Formato: `{"objeto": "[valor1, valor2, valor3]"}`

3. **Gráficos** (PNG)
   - Localização: `output/figure/`
   - Visualizações comparativas com valores reais e previstos

### Exemplo de Gráfico

Os gráficos gerados incluem:

- 📈 Valores históricos (9 meses anteriores) - *apenas gráficos de 3 meses*
- 📉 Valores previstos (3 ou 12 meses futuros)
- 🎨 Comparação visual entre Ilhéus e Itabuna
- 🏷️ Anotações com valores exatos
- 📐 Ajuste automático de escala do eixo Y (evita excesso de marcações)
- 🎯 Títulos informativos com período e região

### Otimizações de Código

**Redução de código repetido:**

- ~120 linhas eliminadas através de funções reutilizáveis
- Funções centralizadas em `utils/chart_utils.py`
- Configurações visuais globais (cores, marcadores, tamanhos)
- Lógica inteligente de formatação de eixos

## 🛍️ Produtos Analisados

O sistema analisa os seguintes produtos da cesta básica:

| Produto  | Quantidade | Produto   | Quantidade |
|----------|-----------|-----------|-----------|
| Açúcar   | 3.0 kg    | Leite     | 6.0 L     |
| Arroz    | 3.6 kg    | Manteiga  | 0.75 kg   |
| Banana   | 7.5 kg    | Óleo      | 1.0 L     |
| Café     | 0.3 kg    | Pão       | 6.0 kg    |
| Carne    | 4.5 kg    | Tomate    | 12.0 kg   |
| Farinha  | 3.0 kg    |           |           |
| Feijão   | 4.5 kg    |           |           |

## 📝 Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Faça um Fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença especificada no arquivo LICENSE.

## 👤 Autor

**Matheus Silva**

- GitHub: [@matheusssilva991](https://github.com/matheusssilva991)

## � Documentação Adicional

- [src/utils/README.md](src/utils/README.md) - Documentação completa das funções utilitárias
- [src/models/neural_networks.py](src/models/neural_networks.py) - Arquiteturas dos modelos

## �🙏 Agradecimentos

- Dados coletados das cidades de Ilhéus e Itabuna, Bahia
- Comunidade Python e TensorFlow/Keras

---

⭐ Se este projeto foi útil para você, considere dar uma estrela no repositório!
