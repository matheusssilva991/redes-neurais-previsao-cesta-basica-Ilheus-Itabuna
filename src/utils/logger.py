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
    """Configura um logger do projeto com saída para stdout.

    Substitui os handlers diretos do logger para evitar mensagens duplicadas
    quando a configuração é chamada novamente com o mesmo nome.

    Args:
        name: Nome usado para obter ou criar o logger.
        level: Nível mínimo de log, como `logging.INFO`.
        log_file: Caminho opcional de um arquivo UTF-8 em modo de acréscimo.

    Returns:
        Logger configurado.

    Raises:
        OSError: Se não for possível criar o diretório ou abrir o arquivo de log.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Logger padrão do projeto
default_logger = setup_logger()
