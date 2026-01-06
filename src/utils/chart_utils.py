"""
Utility functions for chart generation.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path


def load_forecast_from_json(filepath):
    """
    Load forecast results from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        list: List of forecast values (multiplied by 1000)
    """
    with open(filepath, "r") as file:
        data = json.load(file)
        key = list(data.keys())[0]

        # Clean and parse the string representation of list
        values_str = data[key].replace("[", "").replace("]", "")
        values = [float(v) * 1000 for v in values_str.split(",")]

        return values


def load_forecasts_for_regions(model, object_name, regions, forecast_horizon=3):
    """
    Load forecasts for multiple regions.

    Args:
        model: Model name (RNN, LSTM, CNN)
        object_name: Object name (e.g., 'cesta_basica')
        regions: List of region names
        forecast_horizon: Forecast horizon (3 or 12)

    Returns:
        dict: Dictionary with region names as keys and forecast lists as values
    """
    results = {}

    for region in regions:
        if forecast_horizon == 12:
            filename = f"previsao_{model}_12_meses_{object_name}_{region}.json"
        else:
            filename = f"previsao_{model}_{object_name}_{region}.json"

        filepath = f"../../output/previsoes_cesta/{filename}"
        results[region] = load_forecast_from_json(filepath)

    return results


def load_product_forecasts(model, regions, products, quantities, forecast_horizon=12):
    """
    Load forecasts for multiple products and regions.

    Args:
        model: Model name
        regions: List of region names
        products: List of product names
        quantities: Dictionary with product quantities
        forecast_horizon: Forecast horizon

    Returns:
        dict: Nested dictionary {region: {product: [values]}}
    """
    results = {region: {} for region in regions}

    for region in regions:
        for product in products:
            if forecast_horizon == 12:
                filename = f"previsao_{model}_12_meses_{product}_{region}.json"
            else:
                filename = f"previsao_{model}_{product}_{region}.json"

            filepath = f"../../output/previsoes_produtos/{region}/{filename}"

            with open(filepath, "r") as file:
                data = json.load(file)
                values_str = data[product].replace("[", "").replace("]", "")
                values = [float(v) for v in values_str.split(",")]

                # Normalize by quantity
                values = [v / quantities[product] * 1000 for v in values]
                results[region][product] = values

    return results


def setup_plot_style(figsize=(15, 5)):
    """
    Create and configure a matplotlib figure.

    Args:
        figsize: Tuple with figure size

    Returns:
        tuple: (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def format_yticks_with_comma(ax, yticks_range, yticks_labels=None):
    """
    Format Y-axis ticks with comma as decimal separator.

    Args:
        ax: Matplotlib axis
        yticks_range: Range or list of tick positions
        yticks_labels: Optional custom labels
    """
    if yticks_labels is None:
        yticks_labels = [format(y, ".2f").replace(".", ",") for y in yticks_range]

    ax.set_yticks(yticks_range)
    ax.set_yticklabels(yticks_labels, size=15)


def add_forecast_annotation(ax, x, y, value, offset=5, va="bottom"):
    """
    Add annotation to a forecast point.

    Args:
        ax: Matplotlib axis
        x: X coordinate
        y: Y coordinate
        value: Value to display
        offset: Vertical offset
        va: Vertical alignment ('top' or 'bottom')
    """
    ax.annotate(
        format(value, ".2f").replace(".", ","),
        (x, y + offset if va == "bottom" else y - offset),
        size=13,
        va=va,
        ha="center",
    )


def save_figure(filename, dpi=300):
    """
    Save current matplotlib figure.

    Args:
        filename: Output filename
        dpi: Resolution in dots per inch
    """
    plt.savefig(fname=filename, dpi=dpi, bbox_inches="tight")
    print(f"✅ Gráfico salvo em: {filename}")


def load_product_historical_data(
    data_folder, sub_pasta, region, products, quantities, n_months=9
):
    """
    Load historical product data from Excel files.

    Args:
        data_folder: Base data folder path
        sub_pasta: Subfolder name
        region: Region name
        products: List of product names
        quantities: Dictionary with product quantities
        n_months: Number of last months to load

    Returns:
        dict: Dictionary with product names as keys and historical values as lists
    """
    import pandas as pd

    path = Path(data_folder.encode("latin1").decode("utf-8"))
    data_path = path / sub_pasta

    results = {}
    for product in products:
        file_path = data_path / f"{product}_{region}.xlsx"
        df = pd.read_excel(file_path)
        values = df["preco"].tail(n_months).to_list()
        # Normalize by quantity
        values = [v / quantities[product] for v in values]
        results[product] = values

    return results


def plot_product_chart(
    ax,
    produtos,
    valores_reais,
    previsoes,
    markers,
    colors_real,
    colors_prev,
    markers_size,
    line_size,
    x_label,
    limite_y,
):
    """
    Plot product chart with historical values and forecasts.

    Args:
        ax: Matplotlib axis
        produtos: List of product names
        valores_reais: Dictionary with historical values per product
        previsoes: Dictionary with forecast values per product
        markers: List of marker styles
        colors_real: List of colors for historical values
        colors_prev: List of colors for forecasts
        markers_size: List of marker sizes
        line_size: Line width
        x_label: X-axis labels
        limite_y: Y-axis upper limit
    """
    for i, produto in enumerate(produtos):
        # Connection line
        ax.plot(
            [8, 9],
            [valores_reais[produto][-1], previsoes[produto][0]],
            ":",
            color=colors_prev[i],
            lw=line_size,
            markersize=markers_size[i],
        )

        # Historical values
        ax.plot(
            [x for x in range(9)],
            valores_reais[produto],
            marker=markers[i],
            color=colors_real[i],
            lw=line_size,
            markersize=markers_size[i],
        )

        # Forecasts
        ax.plot(
            [9, 10, 11],
            previsoes[produto],
            ":",
            marker=markers[i],
            label=produto.title().replace("c", "ç"),
            color=colors_prev[i],
            lw=line_size,
            markersize=markers_size[i],
        )

    # Formatting
    ax.set_xticks([x for x in range(0, 12)])
    ax.set_xticklabels(x_label, size=15)
    ax.set_ylabel("Custo (R$)", size=20)
    ax.set_yticks([y for y in range(0, limite_y, 10)])
    ax.set_yticklabels(
        [format(y, ".2f").replace(".", ",") for y in range(0, limite_y, 10)], size=15
    )
    ax.grid(which="major", axis="y", alpha=0.5)


def plot_forecast_only_chart(
    ax,
    produtos,
    previsoes,
    markers,
    colors,
    markers_size,
    line_size,
    x_labels,
    yticks_range,
    yticks_labels,
):
    """
    Plot product chart with only forecast values (no historical data).

    Args:
        ax: Matplotlib axis
        produtos: List of product names
        previsoes: Dictionary with forecast values per product
        markers: List of marker styles
        colors: List of colors for forecasts
        markers_size: List of marker sizes
        line_size: Line width
        x_labels: X-axis labels
        yticks_range: Y-axis tick positions
        yticks_labels: Y-axis tick labels
    """
    for produto, mark, color, marker_size in zip(
        produtos, markers, colors, markers_size
    ):
        ax.plot(
            range(12),
            previsoes[produto],
            label=produto.title().replace("c", "ç"),
            color=color,
            marker=mark,
            ls="--",
            markersize=marker_size,
            lw=line_size,
        )

    # Formatting
    ax.set_xticks(range(12))
    ax.set_xticklabels(x_labels, size=13)
    ax.set_yticks(yticks_range)
    ax.set_yticklabels(yticks_labels, size=13)
    ax.set_ylabel("Preço dos produtos da cesta básica (R$)", size=18)
    ax.grid(which="major", axis="y", alpha=0.5)
