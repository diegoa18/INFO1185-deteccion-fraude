"""Diagnóstico y visualización del desbalance de clases."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import TARGET_COLUMN


def analyze_class_imbalance(
    target: pd.Series,
    *,
    target_name: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Resume clases 0/1, porcentajes y razón mayoría/minoría.

    La clase ``1`` se valida y se denomina siempre ``fraude``. Una accuracy alta
    por sí sola sería engañosa: con una minoría tan pequeña, predecir siempre la
    clase 0 obtiene una accuracy alta pero recall de fraude igual a cero.
    """
    if target.empty:
        raise ValueError("No es posible analizar un objetivo vacío.")
    counts = target.value_counts().reindex([0, 1], fill_value=0).astype(int)
    if counts.loc[1] == 0:
        raise ValueError("El objetivo debe contener al menos un fraude (Class == 1).")
    if counts.loc[0] == 0:
        raise ValueError("El objetivo debe contener al menos una transacción legítima.")

    majority_to_minority = counts.max() / counts.min()
    summary = pd.DataFrame(
        {
            "class": counts.index,
            "label": ["legítima", "fraude"],
            "count": counts.to_numpy(),
            "percentage": (counts / counts.sum() * 100).to_numpy(),
            "majority_to_minority_ratio": majority_to_minority,
        }
    )
    summary.attrs["target_name"] = target_name
    return summary


def plot_class_imbalance(
    imbalance_summary: pd.DataFrame,
    output_path: str | Path | None = None,
):
    """Crea una visualización simple y opcional de la distribución de clases."""
    # La importación diferida evita inicializar Matplotlib cuando solo se ejecuta
    # la optimización o el análisis tabular en un entorno sin interfaz gráfica.
    import matplotlib.pyplot as plt

    required = {"class", "label", "count"}
    if not required.issubset(imbalance_summary.columns):
        raise ValueError("imbalance_summary no contiene las columnas requeridas.")

    figure, axis = plt.subplots(figsize=(6, 4))
    bars = axis.bar(
        imbalance_summary["class"].astype(str),
        imbalance_summary["count"],
        color=["#4C78A8", "#E45756"],
    )
    axis.set(
        title="Distribución de clases: transacciones legítimas y fraude",
        xlabel="Clase (0: legítima, 1: fraude)",
        ylabel="Cantidad de transacciones",
    )
    axis.bar_label(bars, labels=[f"{count:,}" for count in imbalance_summary["count"]])
    figure.tight_layout()

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150, bbox_inches="tight")
    return figure, axis
