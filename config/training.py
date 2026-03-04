"""
Configurações de treinamento: hiperparâmetros padrão para modelos.
"""

# Parâmetros de série temporal
DEFAULT_LOOK_BACK = 12  # Janela temporal de entrada (meses)
DEFAULT_FORECAST_HORIZON = 3  # Horizonte de previsão padrão (meses)

# Hiper parâmetros de treinamento
DEFAULT_EPOCHS = 150  # Número de épocas
DEFAULT_BATCH_SIZE = 1  # Tamanho do batch
DEFAULT_LEARNING_RATE = 0.0003  # Taxa de aprendizado
DEFAULT_NORMALIZATION_FACTOR = 1000  # Fator de normalização
