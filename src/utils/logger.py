"""
Utilitário de logging estruturado para o projeto.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "previsao_cestas",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Configura um logger para o projeto.

    Args:
        name: Nome do logger
        level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Caminho opcional para arquivo de log

    Returns:
        logging.Logger: Logger configurado

    Examples:
        >>> logger = setup_logger()
        >>> logger.info("Modelo treinado com sucesso")
        >>> logger.error("Erro ao carregar dados")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove handlers existentes para evitar duplicação
    logger.handlers.clear()

    # Formato do log
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler para arquivo (opcional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Logger padrão do projeto
default_logger = setup_logger()
