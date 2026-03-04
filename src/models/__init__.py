"""
Módulo de modelos de redes neurais para previsão de séries temporais.

Disponibiliza:
- get_model(): Factory function para criar modelos
- AVAILABLE_MODELS: Dicionário com modelos disponíveis
- create_rnn_model(), create_lstm_model(), create_cnn_model(): Construtores específicos

Uso:
    from models import get_model

    model = get_model('RNN', look_back=12, forecast_horizon=3)
    model.summary()
"""

from .neural_networks import (
    get_model,
    AVAILABLE_MODELS,
    create_rnn_model,
    create_lstm_model,
    create_cnn_model,
)

__all__ = [
    "get_model",
    "AVAILABLE_MODELS",
    "create_rnn_model",
    "create_lstm_model",
    "create_cnn_model",
]
