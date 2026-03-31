#!/usr/bin/env python3
"""
Interface de Linha de Comando (CLI) para treinamento de modelos de previsão.

Este script permite treinar modelos de previsão via terminal, oferecendo
uma alternativa ao Jupyter Notebook para automação e integração CI/CD.

Uso básico:
    python src/cli.py train --model RNN --horizon 3 --region ilheus

Para ajuda:
    python src/cli.py --help
"""

from typing import List
from pathlib import Path
import sys

# Adicionar src ao path para importações
sys.path.insert(0, str(Path(__file__).parent))

try:
    import click
except ImportError:
    print("❌ Erro: pacote 'click' não instalado.")
    print("Instale com: pip install click")
    sys.exit(1)

from models import get_model
from utils import (
    load_data,
    create_time_sequences,
    prepare_training_data,
    train_model,
    save_model,
    generate_forecast,
    save_forecasts,
)
from utils.logger import setup_logger
from config import (
    REGIOES,
    PRODUTOS,
    DEFAULT_LOOK_BACK,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DATA_DIR,
    PROJECT_ROOT,
)

logger = setup_logger(__name__)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    🛒 Sistema de Previsão de Preços da Cesta Básica

    Ferramenta CLI para treinar modelos de Deep Learning e gerar
    previsões de preços para Ilhéus e Itabuna.
    """
    pass


@cli.command()
@click.option(
    "--model",
    "-m",
    type=click.Choice(["RNN", "LSTM", "CNN"], case_sensitive=True),
    default="RNN",
    help="Modelo de rede neural a usar",
)
@click.option(
    "--horizon",
    "-h",
    type=click.Choice(["3", "6", "12"]),
    default="3",
    help="Horizonte de previsão em meses",
)
@click.option(
    "--region",
    "-r",
    type=click.Choice(["ilheus", "itabuna", "both"]),
    default="both",
    help="Região para treinar (both = ambas)",
)
@click.option(
    "--forecast-type",
    "-t",
    type=click.Choice(["cesta", "produtos", "both"]),
    default="cesta",
    help="Tipo de previsão: cesta básica ou produtos individuais",
)
@click.option(
    "--epochs",
    "-e",
    type=int,
    default=DEFAULT_EPOCHS,
    help=f"Número de épocas de treinamento (padrão: {DEFAULT_EPOCHS})",
)
@click.option(
    "--learning-rate",
    "-lr",
    type=float,
    default=DEFAULT_LEARNING_RATE,
    help=f"Taxa de aprendizado (padrão: {DEFAULT_LEARNING_RATE})",
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Mostrar progresso detalhado do treinamento"
)
@click.option(
    "--save-onnx",
    is_flag=True,
    help="Exportar modelos também em formato ONNX (.onnx)",
)
def train(
    model: str,
    horizon: str,
    region: str,
    forecast_type: str,
    epochs: int,
    learning_rate: float,
    verbose: bool,
    save_onnx: bool,
):
    """
    Treina modelos de previsão e gera previsões.

    Exemplos:

        # Treinar RNN para cesta básica de ambas regiões (3 meses)
        $ python src/cli.py train --model RNN --horizon 3 --region both

        # Treinar LSTM para produtos de Ilhéus (12 meses)
        $ python src/cli.py train -m LSTM -h 12 -r ilheus -t produtos

        # CNN com configurações customizadas
        $ python src/cli.py train -m CNN -e 200 -lr 0.001 -v
    """
    forecast_horizon = int(horizon)
    regions = REGIOES if region == "both" else [region]

    click.echo(f"\n{'=' * 70}")
    click.echo("🚀 INICIANDO TREINAMENTO")
    click.echo(f"{'=' * 70}")
    click.echo(f"Modelo: {model}")
    click.echo(f"Horizonte: {forecast_horizon} meses")
    click.echo(f"Regiões: {', '.join(regions)}")
    click.echo(f"Tipo: {forecast_type}")
    click.echo(f"Épocas: {epochs}")
    click.echo(f"Learning Rate: {learning_rate}")
    click.echo(f"Salvar ONNX: {'sim' if save_onnx else 'não'}")
    click.echo(f"{'=' * 70}\n")

    try:
        if forecast_type in ["cesta", "both"]:
            _train_cesta_basica(
                model,
                regions,
                forecast_horizon,
                epochs,
                learning_rate,
                verbose,
                save_onnx,
            )

        if forecast_type in ["produtos", "both"]:
            _train_produtos(
                model,
                regions,
                forecast_horizon,
                epochs,
                learning_rate,
                verbose,
                save_onnx,
            )

        click.echo(f"\n{'=' * 70}")
        click.echo("🎉 TREINAMENTO CONCLUÍDO COM SUCESSO!")
        click.echo(f"{'=' * 70}\n")

    except Exception as e:
        logger.error(f"Erro durante treinamento: {e}", exc_info=True)
        click.echo(f"\n❌ Erro: {e}", err=True)
        sys.exit(1)


def _train_cesta_basica(
    model_name: str,
    regions: List[str],
    forecast_horizon: int,
    epochs: int,
    learning_rate: float,
    verbose: bool,
    save_onnx: bool,
):
    """Treina modelos para previsão da cesta básica."""
    click.echo("\n📦 TREINANDO CESTA BÁSICA")
    click.echo(f"{'=' * 70}\n")

    for regiao in regions:
        click.echo(f"📍 Região: {regiao.upper()}")

        # Carregar dados
        data_path = DATA_DIR / f"accb_custo_total_{regiao}.xlsx"

        if not data_path.exists():
            click.echo(f"  ⚠️  Arquivo não encontrado: {data_path}", err=True)
            continue

        df = load_data(data_path)
        df = create_time_sequences(df, DEFAULT_LOOK_BACK, forecast_horizon)
        X_train, y_train, X_val = prepare_training_data(df, DEFAULT_LOOK_BACK)

        # Criar e treinar modelo
        model = get_model(
            model_name, DEFAULT_LOOK_BACK, forecast_horizon, learning_rate
        )

        with click.progressbar(
            length=epochs, label=f"  Treinando {model_name}", show_percent=True
        ) as bar:
            train_model(
                model,
                X_train,
                y_train,
                epochs=epochs,
                batch_size=DEFAULT_BATCH_SIZE,
                verbose=1 if verbose else 0,
            )
            bar.update(epochs)

        # Salvar modelo
        save_model(
            model,
            model_name,
            regiao,
            "cesta_basica",
            forecast_horizon,
            save_onnx=save_onnx,
        )

        # Gerar e salvar previsões
        results = generate_forecast(model, X_val, batch_size=DEFAULT_BATCH_SIZE)
        save_forecasts(
            results,
            model_name,
            "cesta_basica",
            regiao,
            forecast_horizon,
            forecast_type="cesta",
        )

        # Mostrar previsões
        results_reais = [r * 1000 for r in results]
        click.echo("  ✅ Treinado e salvo")
        click.echo(f"  📊 Previsões: {[round(r, 2) for r in results_reais]}\n")


def _train_produtos(
    model_name: str,
    regions: List[str],
    forecast_horizon: int,
    epochs: int,
    learning_rate: float,
    verbose: bool,
    save_onnx: bool,
):
    """Treina modelos para previsão de produtos individuais."""
    click.echo("\n🛍️  TREINANDO PRODUTOS INDIVIDUAIS")
    click.echo(f"{'=' * 70}\n")
    click.echo(
        f"Total: {len(regions)} regiões × {len(PRODUTOS)} produtos = {len(regions) * len(PRODUTOS)} modelos\n"
    )

    total = len(regions) * len(PRODUTOS)

    with click.progressbar(
        length=total, label="Progresso geral", show_percent=True, show_pos=True
    ) as bar:
        for regiao in regions:
            click.echo(f"\n📍 Região: {regiao.upper()}")

            for produto in PRODUTOS:
                # Carregar dados
                data_path = (
                    DATA_DIR / "datasets_produtos" / regiao / f"{produto}_{regiao}.xlsx"
                )

                if not data_path.exists():
                    click.echo(f"  ⚠️  {produto}: arquivo não encontrado", err=True)
                    bar.update(1)
                    continue

                df = load_data(data_path)
                df = create_time_sequences(df, DEFAULT_LOOK_BACK, forecast_horizon)
                X_train, y_train, X_val = prepare_training_data(df, DEFAULT_LOOK_BACK)

                # Criar e treinar modelo
                model = get_model(
                    model_name, DEFAULT_LOOK_BACK, forecast_horizon, learning_rate
                )
                train_model(
                    model,
                    X_train,
                    y_train,
                    epochs=epochs,
                    batch_size=DEFAULT_BATCH_SIZE,
                    verbose=1 if verbose else 0,
                )

                # Salvar modelo
                save_model(
                    model,
                    model_name,
                    regiao,
                    produto,
                    forecast_horizon,
                    subdir="produtos",
                    save_onnx=save_onnx,
                )

                # Gerar e salvar previsões
                results = generate_forecast(model, X_val, batch_size=DEFAULT_BATCH_SIZE)
                save_forecasts(
                    results,
                    model_name,
                    produto,
                    regiao,
                    forecast_horizon,
                    forecast_type="produtos",
                )

                click.echo(f"  ✅ {produto.capitalize()}")
                bar.update(1)


@cli.command()
@click.option(
    "--region",
    "-r",
    type=click.Choice(["ilheus", "itabuna", "both"]),
    default="both",
    help="Região para listar modelos",
)
def list_models(region: str):
    """
    Lista modelos treinados disponíveis.

    Exemplo:
        $ python src/cli.py list-models --region ilheus
    """
    from config import MODELS_DIR

    regions = REGIOES if region == "both" else [region]

    click.echo("\n📁 MODELOS TREINADOS")
    click.echo(f"{'=' * 70}\n")

    for reg in regions:
        region_dir = MODELS_DIR / reg

        if not region_dir.exists():
            click.echo(f"📍 {reg.upper()}: Nenhum modelo encontrado\n")
            continue

        # Listar modelos da cesta básica
        cesta_models = list(region_dir.glob("*cesta_basica*.keras"))

        # Listar modelos de produtos
        produtos_dir = region_dir / "produtos"
        produto_models = []
        if produtos_dir.exists():
            produto_models = list(produtos_dir.glob("*.keras"))

        click.echo(f"📍 {reg.upper()}")
        click.echo(f"   Cesta Básica: {len(cesta_models)} modelo(s)")
        for model in cesta_models:
            click.echo(f"     • {model.name}")

        if produto_models:
            click.echo(f"   Produtos: {len(produto_models)} modelo(s)")
            for model in produto_models[:5]:  # Mostrar apenas primeiros 5
                click.echo(f"     • {model.name}")
            if len(produto_models) > 5:
                click.echo(f"     ... e mais {len(produto_models) - 5}")

        click.echo()


@cli.command()
def info():
    """
    Mostra informações sobre o projeto e configurações.
    """
    import yaml

    click.echo(f"\n{'=' * 70}")
    click.echo("ℹ️  INFORMAÇÕES DO PROJETO")
    click.echo(f"{'=' * 70}\n")

    click.echo(f"📂 Diretório do projeto: {PROJECT_ROOT}")
    click.echo(f"📊 Regiões: {', '.join(REGIOES)}")
    click.echo(f"🛍️  Produtos: {len(PRODUTOS)} produtos")
    click.echo("📈 Modelos disponíveis: RNN, LSTM, CNN")
    click.echo("⏱️  Horizontes de previsão: 3, 6, 12 meses")

    # Tentar ler configurações do YAML
    config_file = PROJECT_ROOT / "config" / "models.yaml"
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                click.echo(f"\n📝 Configuração carregada de: {config_file}")
                click.echo(
                    f"   Look back: {config['training']['default']['look_back']} meses"
                )
                click.echo(
                    f"   Épocas padrão: {config['training']['default']['epochs']}"
                )
        except Exception as e:
            click.echo(f"\n⚠️  Erro ao ler configuração: {e}", err=True)

    click.echo(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    cli()
