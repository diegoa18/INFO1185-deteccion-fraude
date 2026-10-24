"""Genera figuras de la presentación a partir de los resultados versionados."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
OUTPUT = Path(__file__).resolve().parent / "figures"

BLUE = "#173F5F"
CYAN = "#2A9D8F"
ORANGE = "#F4A261"
RED = "#E76F51"
GRAY = "#64748B"


def _setup() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)


def class_distribution() -> None:
    """Grafica el desbalance de la muestra experimental en escala logarítmica."""
    data = pd.read_csv(TABLES / "issue_7_class_imbalance.csv")
    labels = ["Legítimas" if value == 0 else "Fraudes" for value in data["class"]]
    colors = [BLUE, RED]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.bar(labels, data["count"], color=colors, width=0.58)
    ax.set_yscale("log")
    ax.set_ylim(35, 55_000)
    ax.set_ylabel("Transacciones (escala logarítmica)")
    ax.set_title("Muestra experimental: 30.000 transacciones", pad=14)
    ax.grid(axis="x", visible=False)
    for bar, row in zip(bars, data.itertuples(index=False)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.08,
            f"{int(row.count):,}\n({row.percentage:.3f} %)".replace(",", "."),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax.text(
        0.98,
        0.48,
        "Razón ≈ 576 : 1",
        transform=ax.transAxes,
        ha="right",
        color=GRAY,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / "class_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _condition_labels(data: pd.DataFrame) -> list[str]:
    abbreviations = {
        "k-NN": "k-NN",
        "SVM": "SVM",
        "Árbol de Decisión": "Árbol",
        "Random Forest propio": "RF propio",
    }
    return [
        f"{abbreviations[row.model]} · {'con ROS' if row.balanced else 'sin ROS'}"
        for row in data.itertuples(index=False)
    ]


def cv_tradeoff() -> None:
    """Compara sensibilidad y precisión medias de validación cruzada."""
    data = pd.read_csv(TABLES / "issue_8_validation_metrics.csv")
    labels = _condition_labels(data)
    positions = np.arange(len(data))
    height = 0.34
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.barh(
        positions - height / 2,
        data["sensitivity_cv_mean"],
        height,
        xerr=data["sensitivity_cv_std"],
        color=CYAN,
        label="Sensibilidad media ± σ",
        capsize=3,
    )
    ax.barh(
        positions + height / 2,
        data["precision_cv_mean"],
        height,
        xerr=data["precision_cv_std"],
        color=ORANGE,
        label="Precisión media ± σ",
        capsize=3,
    )
    selected = data.index[(data["model"] == "SVM") & data["balanced"]].item()
    ax.axhspan(selected - 0.48, selected + 0.48, color=RED, alpha=0.10)
    ax.text(0.99, selected, "seleccionado", ha="right", va="center", color=RED, fontsize=11)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Métrica en validación cruzada")
    ax.set_title("Selección: sensibilidad primero, precisión para desempatar")
    ax.legend(loc="lower right", frameon=True, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUT / "cv_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def test_tradeoff() -> None:
    """Compara sensibilidad, precisión y F1 de las ocho condiciones en prueba."""
    data = pd.read_csv(TABLES / "issue_8_test_metrics.csv")
    labels = _condition_labels(data)
    positions = np.arange(len(data))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.barh(positions - width, data["sensitivity_test"], width, color=CYAN, label="Sensibilidad")
    ax.barh(positions, data["precision_test"], width, color=ORANGE, label="Precisión")
    ax.barh(positions + width, data["f1_score_test"], width, color=BLUE, label="F1-score")
    selected = data.index[(data["model"] == "SVM") & data["balanced"]].item()
    ax.axhspan(selected - 0.47, selected + 0.47, color=RED, alpha=0.10)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Métrica puntual en test reservado")
    ax.set_title("Resultado final: detectar más fraude puede elevar las falsas alarmas")
    ax.legend(loc="lower right", ncols=3, frameon=True, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUT / "test_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def selected_confusions() -> None:
    """Muestra el costo operativo del modelo seleccionado frente a k-NN base."""
    data = pd.read_csv(TABLES / "issue_8_confusion_matrices.csv")
    conditions = [
        ("k-NN", False, "k-NN sin balanceo"),
        ("SVM", True, "SVM con balanceo\n(seleccionado por CV)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4))
    for ax, (model, balanced, title) in zip(axes, conditions):
        row = data.loc[(data["model"] == model) & (data["balanced"] == balanced)].iloc[0]
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues" if not balanced else "Oranges",
            cbar=False,
            square=True,
            linewidths=1,
            xticklabels=["Legítima", "Fraude"],
            yticklabels=["Legítima", "Fraude"],
            annot_kws={"fontsize": 15, "fontweight": "bold"},
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Clase real")
    fig.suptitle("Matrices de confusión en test (9.000 transacciones)", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT / "selected_confusions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _setup()
    class_distribution()
    cv_tradeoff()
    test_tradeoff()
    selected_confusions()


if __name__ == "__main__":
    main()
