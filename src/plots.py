"""Generación de figuras con estilo consistente para el informe y la portada."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config

PALETTE = ["#0B5394", "#E69138", "#C0392B", "#27AE60", "#7F8C8D"]
sns.set_theme(style="whitegrid", palette=PALETTE, font="DejaVu Sans")
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (9, 5.5),
    }
)


def _save(fig: plt.Figure, name: str, folder: str) -> str:
    path = (config.FIGURES_EDA if folder == "eda" else config.FIGURES_MODEL) / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
def eda_bad_rate_by_grade(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots()
    order = ["A", "B", "C", "D", "E", "F", "G"]
    rates = [
        df.loc[df["grade"] == g, "bad_loan"].mean() if (df["grade"] == g).sum() else 0
        for g in order
    ]
    counts = [(df["grade"] == g).sum() for g in order]
    bars = ax.bar(order, [r * 100 for r in rates], color=PALETTE[0], alpha=0.9)
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"n={c:,}",
            ha="center",
            fontsize=8,
            color="#444444",
        )
    ax.set_ylim(0, max(rates) * 100 * 1.25 + 2)
    ax.set_xlabel("Calificación asignada por Lending Club (grade)")
    ax.set_ylabel("Tasa de mal préstamo (%)")
    ax.set_title("Tasa observada de mal préstamo por calificación crediticia")
    return _save(fig, "eda_bad_rate_by_grade", "eda")


def eda_delinq_profile(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bins = [0, 1, 2, 3, 5, 11]
    labels = ["0", "1", "2", "3-4", "5+"]
    df["delinq_cat"] = pd.cut(
        df["delinq_2yrs"], bins=bins, labels=labels, right=False, include_lowest=True
    )
    rate = df.groupby("delinq_cat", observed=True)["bad_loan"].mean() * 100
    count = df.groupby("delinq_cat", observed=True).size()
    ax = axes[0]
    ax.bar(rate.index.astype(str), rate.values, color=PALETTE[1], alpha=0.95)
    ax.set_title("Tasa de mal préstamo por # morosidades (2 años)")
    ax.set_xlabel("Morosidades en los últimos 2 años")
    ax.set_ylabel("Tasa de mal préstamo (%)")

    ax = axes[1]
    m = df["mths_since_last_delinq"].dropna()
    _ = ax.hist(
        m[m <= 120], bins=40, color=PALETTE[4], alpha=0.9, edgecolor="white"
    )
    ax.set_title("Distribución de meses desde última morosidad")
    ax.set_xlabel("Meses desde la última morosidad (<= 120)")
    ax.set_ylabel("Frecuencia")
    fig.suptitle("Perfil de morosidad de la cartera", fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "eda_delinq_profile", "eda")


def eda_income_dti(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sample = df.sample(min(60_000, len(df)), random_state=config.RANDOM_STATE)
    ax = axes[0]
    ax.scatter(
        sample["annual_inc"],
        sample["dti"],
        s=4,
        alpha=0.25,
        c=np.where(sample["bad_loan"] == 1, PALETTE[2], PALETTE[0]),
    )
    ax.set_xlim(0, 250_000)
    ax.set_xlabel("Ingreso anual declarado (USD)")
    ax.set_ylabel("Relación deuda-ingreso (dti)")
    ax.set_title("Ingreso vs. DTI (rojo = mal préstamo)")

    ax = axes[1]
    ax.hist(
        sample.loc[sample["bad_loan"] == 1, "dti"],
        bins=30,
        alpha=0.55,
        color=PALETTE[2],
        label="Mal préstamo",
        density=True,
    )
    ax.hist(
        sample.loc[sample["bad_loan"] == 0, "dti"],
        bins=30,
        alpha=0.55,
        color=PALETTE[0],
        label="Buen préstamo",
        density=True,
    )
    ax.set_xlabel("Relación deuda-ingreso (dti)")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución del DTI por condición del préstamo")
    ax.legend()
    fig.suptitle("Capacidad de pago de los solicitantes", fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "eda_income_dti", "eda")


def eda_correlation(df: pd.DataFrame) -> str:
    cols = [
        "bad_loan",
        "grade_num",
        "int_rate",
        "dti",
        "delinq_2yrs",
        "mths_since_last_delinq",
        "credit_history_months",
        "annual_inc",
        "revol_util",
        "open_acc",
        "collections_12_mths_ex_med",
    ]
    corr = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Correlación de Spearman"},
        ax=ax,
        annot_kws={"fontsize": 8},
    )
    ax.set_title("Matriz de correlación de variables del modelo")
    return _save(fig, "eda_correlation", "eda")


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def plot_roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray, float]]) -> str:
    """curves: {etiqueta: (fpr, tpr, auc)}. La referencia aleatoria se agrega sola."""
    fig, ax = plt.subplots()
    for label, (fpr, tpr, auc_v) in curves.items():
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {auc_v:.4f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#7F8C8D", lw=1, label="Clasificación aleatoria")
    ax.set_xlabel("Tasa de falsos positivos (1 - especificidad)")
    ax.set_ylabel("Tasa de verdaderos positivos (sensibilidad)")
    ax.set_title("Curvas ROC del score de riesgo")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    return _save(fig, "model_roc", "model")


def plot_score_distribution(
    score_bueno: np.ndarray,
    score_malo: np.ndarray,
    cutoff_aprobacion: float,
    cutoff_interno: float,
) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.hist(
        score_bueno,
        bins=60,
        alpha=0.55,
        color=PALETTE[0],
        label="Buen préstamo (recuperación ≥ 70%)",
        density=True,
    )
    ax.hist(
        score_malo,
        bins=60,
        alpha=0.55,
        color=PALETTE[2],
        label="Mal préstamo (recuperación < 70%)",
        density=True,
    )
    ylim = ax.get_ylim()
    ax.axvline(cutoff_aprobacion, color=PALETTE[1], lw=2, ls="--")
    ax.text(
        cutoff_aprobacion + 2,
        ylim[1] * 0.94,
        f"Aprobación (score = {cutoff_aprobacion:.0f})",
        color=PALETTE[1],
        fontweight="bold",
        fontsize=9,
    )
    if cutoff_interno > cutoff_aprobacion:
        ax.axvline(cutoff_interno, color=PALETTE[3], lw=2, ls="--")
        ax.text(
            cutoff_interno + 2,
            ylim[1] * 0.86,
            f"Segmento Bueno/Malo (score = {cutoff_interno:.0f})",
            color=PALETTE[3],
            fontweight="bold",
            fontsize=9,
        )
    ax.set_xlabel("Puntaje crediticio (scorecard)")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución del puntaje por condición real del préstamo")
    ax.legend(loc="upper left")
    return _save(fig, "model_score_distribution", "model")


def plot_confusion_matrix(y_true: np.ndarray, aprobado: np.ndarray) -> str:
    cm = np.zeros((2, 2), dtype=int)
    cm[0, 0] = int(((y_true == 0) & (aprobado == 1)).sum())  # bueno, aprobado
    cm[0, 1] = int(((y_true == 0) & (aprobado == 0)).sum())  # bueno, rechazado
    cm[1, 0] = int(((y_true == 1) & (aprobado == 1)).sum())  # malo, aprobado
    cm[1, 1] = int(((y_true == 1) & (aprobado == 0)).sum())  # malo, rechazado
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=1,
        ax=ax,
        annot_kws={"fontsize": 13},
        xticklabels=["Aprobado", "Rechazado"],
        yticklabels=["Bueno (0)", "Malo (1)"],
    )
    ax.set_xlabel("Decisión del modelo")
    ax.set_ylabel("Condición real del préstamo")
    ax.set_title("Matriz de confusión de la política de aprobación (test)")
    return _save(fig, "model_confusion_matrix", "model")


def plot_lift_chart(df: pd.DataFrame, target: str = "bad_loan") -> str:
    """Curva de captura: % de malos capturados por % de cartera revisada."""
    d = df.sort_values("score", ascending=True).reset_index(drop=True)
    malos = (d[target] == 1).cumsum()
    total_malos = malos.iloc[-1]
    pct_cartera = (np.arange(1, len(d) + 1) / len(d)) * 100
    pct_malos = (malos / total_malos) * 100
    fig, ax = plt.subplots()
    ax.plot(pct_cartera, pct_malos, lw=2.2, color=PALETTE[0])
    ax.plot([0, 100], [0, 100], ls="--", lw=1, color="#7F8C8D")
    ax.axvline(25, ls=":", color=PALETTE[1], lw=2)
    ax.text(26, 8, "25% aprobado", color=PALETTE[1], fontweight="bold", fontsize=9)
    ax.set_xlabel("% de solicitudes revisadas (menor score primero)")
    ax.set_ylabel("% de malos préstamos capturados")
    ax.set_title("Curva de captura de malos préstamos (lift)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    return _save(fig, "model_lift_chart", "model")


def plot_approval_rates_by_segment(df: pd.DataFrame) -> str:
    """Tasa de mal préstamo por segmento de la política (evidencia de riesgo)."""
    order = ["Bueno", "Malo", "Aprobado", "Rechazado", "Global"]
    subset = df[df["segmento"].isin(["Bueno", "Malo"])]
    rates = {
        "Bueno": subset.loc[subset["segmento"] == "Bueno", "bad_loan"].mean(),
        "Malo": subset.loc[subset["segmento"] == "Malo", "bad_loan"].mean(),
        "Aprobado": df.loc[df["segmento"].isin(["Bueno", "Malo"]), "bad_loan"].mean(),
        "Rechazado": df.loc[df["segmento"] == "Rechazado", "bad_loan"].mean(),
        "Global": df["bad_loan"].mean(),
    }
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(order, [rates[o] * 100 for o in order], color=PALETTE)
    for bar, val in zip(bars, [rates[o] for o in order]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1%}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylim(0, max(rates.values()) * 100 * 1.25 + 1)
    ax.set_ylabel("Tasa de mal préstamo (%)")
    ax.set_title("Tasa real de mal préstamo por segmento (muestra de test)")
    return _save(fig, "model_approval_rates", "model")